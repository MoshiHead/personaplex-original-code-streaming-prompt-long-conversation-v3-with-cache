# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import random
import os
from pathlib import Path
import tarfile
import time
import traceback
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch
import random

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.diagnostics import diag, _env_flag, _env_int, _IN_SPEECH_RMS, _OUT_SPEECH_RMS
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, other_mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        # Periodic context refresh: the model's RoPE positions are absolute
        # and it collapses into permanent silence once its streaming offset
        # passes the maximum position seen in training (~6100 observed, i.e.
        # ~8 minutes of LM timeline at 12.5 steps/s). Before reaching that
        # cliff, the server rebuilds the context in place: restore the
        # post-prompt KV/streaming state cached once at conversation start
        # (a persistent prompt cache — the voice/text prompts are NOT
        # re-injected through the model, which used to make it greet
        # mid-conversation), then replay only the last
        # ~REFRESH_HISTORY_STEPS of recorded dialogue tokens, batched
        # (covered by a silence keepalive). Offsets below are absolute LM
        # steps.
        self.refresh_enabled = _env_flag("PERSONAPLEX_REFRESH", True)
        self.refresh_soft_offset = _env_int("PERSONAPLEX_REFRESH_SOFT_OFFSET", 5200)
        self.refresh_hard_offset = _env_int("PERSONAPLEX_REFRESH_HARD_OFFSET", 5700)
        self.refresh_quiet_frames = _env_int("PERSONAPLEX_REFRESH_QUIET_FRAMES", 6)
        self.refresh_history_steps = _env_int("PERSONAPLEX_REFRESH_HISTORY_STEPS", 750)
        # Steps per batched transformer forward during replay. Replayed steps
        # are forced on every channel, so they can be batched through the
        # backbone (~batch times faster than per-step: one 7B forward is
        # memory-bandwidth bound regardless of T). <=1 falls back to per-step.
        self.refresh_batch = _env_int("PERSONAPLEX_REFRESH_BATCH", 64)

        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
    
    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize()


    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")
        diag.event("SESSION", f"incoming connection from {peer}:{peer_port}")

        # self.lm_gen.temp = float(request.query["audio_temperature"])
        # self.lm_gen.temp_text = float(request.query["text_temperature"])
        # self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        # self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path
                
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                # Load pre-saved voice prompt embeddings
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(request.query["text_prompt"])) if len(request.query["text_prompt"]) > 0 else None
        seed = int(request["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    diag.beat("recv_loop")
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        diag.event("RECV_LOOP", f"ws error: {ws.exception()!r}", level="error")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        diag.event("RECV_LOOP", "ws CLOSED message received")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        diag.event("RECV_LOOP", "ws CLOSE message received")
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        diag.count("ws_audio_msgs_in")
                        diag.count("ws_audio_bytes_in", len(payload))
                        diag.beat("ws_audio_in")
                        opus_reader.append_bytes(payload)
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
                        diag.count("ws_unknown_msgs_in")
            finally:
                close = True
                clog.log("info", "connection closed")
                diag.event("RECV_LOOP", "exited; close flag set -> other loops will stop")

        async def opus_loop():
            all_pcm_data = None
            frame_idx = 0
            quiet_run = 0  # consecutive frames with neither side speaking

            while True:
                if close:
                    diag.event("OPUS_LOOP", "close flag observed; exiting")
                    return
                await asyncio.sleep(0.001)
                diag.beat("opus_loop")
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                diag.count("pcm_in_samples", int(pcm.shape[-1]))
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    be = time.time()
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    in_rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
                    diag.on_input_frame(in_rms)
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    _t_enc0 = time.monotonic()
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    encode_ms = (time.monotonic() - _t_enc0) * 1000.0
                    for c in range(codes.shape[-1]):
                        _t_step0 = time.monotonic()
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        step_ms = (time.monotonic() - _t_step0) * 1000.0
                        if tokens is None:
                            diag.count("lm_steps_returned_none")
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        _t_dec0 = time.monotonic()
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        decode_ms = (time.monotonic() - _t_dec0) * 1000.0
                        out_np = main_pcm[0, 0].numpy()
                        out_rms = float(np.sqrt(np.mean(np.square(out_np, dtype=np.float64))))
                        opus_writer.append_pcm(out_np)
                        text_token = tokens[0, 0, 0].item()
                        diag.on_output_frame(
                            out_rms=out_rms, text_token=text_token,
                            encode_ms=encode_ms, step_ms=step_ms, decode_ms=decode_ms,
                            frame_wall_ms=(time.time() - be) * 1000.0)
                        frame_idx += 1
                        if diag.enabled and frame_idx % diag.step_log_every == 0:
                            # Deep-dive runs on the inference thread on purpose:
                            # it reads the CUDA-side KV cache offset (device sync).
                            diag.model_stream_summary(self.lm_gen, tokens)
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            diag.on_text_piece(text_token, _text)
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                        else:
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']
                        # ---- Context-refresh trigger ----
                        if refresh["enabled"] and not refresh["in_progress"]:
                            if in_rms < _IN_SPEECH_RMS and out_rms < _OUT_SPEECH_RMS:
                                quiet_run += 1
                            else:
                                quiet_run = 0
                            offset_now = self.lm_gen._streaming_state.offset
                            reason = None
                            if offset_now >= refresh["hard_offset"]:
                                reason = (f"hard deadline: offset {offset_now} at "
                                          f"hard_offset={refresh['hard_offset']} "
                                          f"(position cliff ~6100)")
                            elif (offset_now >= refresh["soft_offset"]
                                  and quiet_run >= refresh["quiet_frames"]):
                                reason = (f"quiet moment ({quiet_run} silent frames) with "
                                          f"offset {offset_now} past "
                                          f"soft_offset={refresh['soft_offset']}")
                            if reason is not None:
                                await do_refresh(reason, offset_now)
                                quiet_run = 0
                                # Drop mic audio that arrived while refreshing: it
                                # is stale by the refresh duration, and stepping
                                # the model through it would burn context steps
                                # and permanently delay playback.
                                all_pcm_data = None
                                stale = opus_reader.read_pcm()
                                if stale.shape[-1] > 0:
                                    diag.event(
                                        "REFRESH",
                                        f"dropped {stale.shape[-1] / self.mimi.sample_rate:.2f}s "
                                        f"of stale mic audio buffered during refresh")
                                break
                    if all_pcm_data is None:
                        # A refresh just ran and the input backlog was dropped;
                        # resume normal frame-by-frame processing.
                        break

        async def send_loop():
            while True:
                if close:
                    diag.event("SEND_LOOP", "close flag observed; exiting")
                    return
                await asyncio.sleep(0.001)
                diag.beat("send_loop")
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    diag.count("ws_audio_bytes_out", len(msg))
                    diag.beat("ws_audio_out")
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if len(request.query["text_prompt"]) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            if seed is not None and seed != -1:
                seed_all(seed)

            diag.register_loop(asyncio.get_running_loop())
            diag.new_conversation({
                "peer": f"{peer}:{peer_port}",
                "voice_prompt": str(voice_prompt_path),
                "text_prompt_chars": len(request.query["text_prompt"]),
                "seed": seed,
            })
            # Arm the watchdog: any of these going stale triggers a warning
            # plus a freeze snapshot with full stack dumps.
            for _name, _timeout in [
                ("recv_loop", 15), ("ws_audio_in", 10), ("audio_in_frame", 10),
                ("opus_loop", 5), ("lm_step", 10), ("audio_out_frame", 10),
                ("ws_audio_out", 10), ("send_loop", 5), ("event_loop", 5),
            ]:
                diag.set_stall_timeout(_name, _timeout)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            self.lm_gen.record_history = False
            self.lm_gen.clear_history()
            # Drop the previous session's prompt cache (frees VRAM; this
            # session snapshots its own state right after its prompts).
            self.lm_gen.clear_prompt_snapshot()
            diag.event("SESSION", "streaming state reset done")
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")

            # ---- Context-refresh controller state ----
            # The model collapses into silence once its absolute offset passes
            # ~6100 (max position seen in training). Refresh rebases positions
            # before that by restoring the cached post-prompt state (see
            # save_prompt_snapshot below) and replaying recent history: soft
            # trigger waits for a quiet moment from soft_offset on; hard
            # trigger fires regardless at hard_offset.
            _prompt_end = self.lm_gen._streaming_state.offset
            refresh = {
                "enabled": self.refresh_enabled,
                "soft_offset": self.refresh_soft_offset,
                "hard_offset": self.refresh_hard_offset,
                "quiet_frames": self.refresh_quiet_frames,
                "in_progress": False,
                "count": 0,
            }
            if refresh["enabled"]:
                # After a refresh the offset restarts at ~prompt_end+history.
                # Make sure that leaves a sane cycle before the next trigger.
                _post = _prompt_end + self.refresh_history_steps
                if refresh["soft_offset"] - _post < 250:
                    refresh["enabled"] = False
                    clog.log("error",
                             f"context refresh disabled: prompt+history ({_post} steps) "
                             f"leaves no room below soft_offset={refresh['soft_offset']}; "
                             f"shorten the text prompt or PERSONAPLEX_REFRESH_HISTORY_STEPS")
                    diag.event("REFRESH", "disabled: prompt+history too long for the "
                               "configured trigger offsets", level="error",
                               prompt_end=_prompt_end,
                               history_steps=self.refresh_history_steps,
                               soft_offset=refresh["soft_offset"])
            if refresh["enabled"]:
                # Persistent prompt cache: deep-copy the post-prompt streaming
                # state (KV caches + offsets) exactly once. Every refresh
                # restores this snapshot in place instead of re-injecting the
                # voice/text prompts through the model, so the system prompt
                # stays active for the whole session and the model never
                # re-experiences a session start (no more mid-conversation
                # greetings), and .wav voice prompts survive refreshes too.
                _t_snap = time.monotonic()
                self.lm_gen.save_prompt_snapshot()
                self.lm_gen.start_history_recording(self.refresh_history_steps + 8)
                clog.log("info",
                         f"context refresh armed: prompt spans steps 0..{_prompt_end} "
                         f"(post-prompt state cached in {time.monotonic() - _t_snap:.2f}s; "
                         f"refreshes restore it instead of re-injecting the prompt), "
                         f"soft trigger at lm_offset>={refresh['soft_offset']}, "
                         f"hard at {refresh['hard_offset']}")
                diag.event("REFRESH", "armed",
                           prompt_end=_prompt_end,
                           prompt_snapshot_cached=self.lm_gen.has_prompt_snapshot,
                           soft_offset=refresh["soft_offset"],
                           hard_offset=refresh["hard_offset"],
                           quiet_frames=refresh["quiet_frames"],
                           history_steps=self.refresh_history_steps,
                           batch=self.refresh_batch)

            async def do_refresh(reason: str, offset_at_trigger: int):
                refresh["in_progress"] = True
                _t0 = time.monotonic()
                clog.log("info", f"context refresh #{refresh['count'] + 1} triggered "
                                 f"at lm_offset={offset_at_trigger}: {reason}")
                diag.event("REFRESH", f"triggered: {reason}",
                           lm_offset=offset_at_trigger,
                           batch=self.refresh_batch,
                           history_steps=self.refresh_history_steps)
                # No output frames are produced while refreshing; silence the
                # watchdog's stall detector so it doesn't dump false snapshots.
                diag.suspend_stall_checks(600)
                keepalive_frame = np.zeros(self.frame_size, dtype=np.float32)
                keepalive_sent = 0

                async def refresh_alive():
                    # The web client force-closes the socket after 10s without
                    # ANY server message (client/src/.../useSocket.ts), so we
                    # must keep real-time-paced audio flowing during the
                    # refresh: silence frames matching the elapsed wall time.
                    # send_loop (still running — we yield below) ships them.
                    # recv_loop owns ws.receive(); never call it from here.
                    nonlocal keepalive_sent
                    target = int((time.monotonic() - _t0) * self.mimi.frame_rate)
                    while keepalive_sent < target:
                        opus_writer.append_pcm(keepalive_frame)
                        keepalive_sent += 1
                        diag.count("refresh_keepalive_frames")
                    diag.beat("opus_loop")
                    diag.beat("audio_out_frame")
                    await asyncio.sleep(0)
                    return not close and not ws.closed

                try:
                    stats = await self.lm_gen.refresh_context_async(
                        is_alive=refresh_alive,
                        batch_size=self.refresh_batch,
                        history_steps=self.refresh_history_steps)
                    refresh["count"] += 1
                    _secs = time.monotonic() - _t0
                    if stats["completed"]:
                        clog.log("info", f"context refresh #{refresh['count']} completed: "
                                         f"offset {offset_at_trigger} -> {stats['offset_after']} "
                                         f"in {_secs:.1f}s")
                        diag.event("REFRESH", "completed successfully",
                                   secs=round(_secs, 2),
                                   keepalive_frames=keepalive_sent,
                                   count=refresh["count"], **stats)
                    else:
                        clog.log("warning",
                                 f"context refresh #{refresh['count']} PARTIAL: "
                                 f"{stats['offset_after']}/{stats['expected_steps']} steps "
                                 f"in {_secs:.1f}s (connection closing)")
                        diag.event("REFRESH", "PARTIAL (aborted by disconnect)",
                                   secs=round(_secs, 2),
                                   keepalive_frames=keepalive_sent,
                                   level="warning", **stats)
                    diag.count("refreshes")
                    diag.gauge("last_refresh_offset", stats["offset_after"])
                except Exception:
                    refresh["enabled"] = False
                    clog.log("error", "context refresh FAILED; disabled for this session")
                    diag.event("REFRESH",
                               f"FAILED, disabled for this session:\n{traceback.format_exc()}",
                               level="error")
                finally:
                    refresh["in_progress"] = False
                    # Give the loops a grace period to refresh their heartbeats
                    # before stall detection resumes.
                    diag.suspend_stall_checks(10)

            async def _traced(name, coro):
                # Task lifecycle tracing: without this, an exception inside any
                # of the session loops is silently swallowed by asyncio.wait.
                diag.event("TASK", f"{name} started")
                try:
                    await coro
                    diag.event("TASK", f"{name} finished normally")
                except asyncio.CancelledError:
                    diag.event("TASK", f"{name} cancelled")
                    raise
                except Exception:
                    diag.event("TASK", f"{name} CRASHED:\n{traceback.format_exc()}",
                               level="error")
                    raise

            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                diag.event("SESSION", "handshake sent; realtime audio streaming begins")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(_traced("recv_loop", recv_loop()), name="recv_loop"),
                    asyncio.create_task(_traced("opus_loop", opus_loop()), name="opus_loop"),
                    asyncio.create_task(_traced("send_loop", send_loop()), name="send_loop"),
                    # Diagnostics only: proves the event loop is alive; never
                    # completes on its own, so FIRST_COMPLETED semantics are unchanged.
                    asyncio.create_task(diag.loop_ticker(), name="diag_ticker"),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    exc = None
                    try:
                        exc = task.exception()
                    except asyncio.CancelledError:
                        pass
                    diag.event("SESSION",
                               f"first completed task: {task.get_name()} exception={exc!r}",
                               level="error" if exc else "info")
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                diag.end_conversation(reason="session loop ended")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
            else:
                diag.end_conversation(reason="client disconnected before handshake")
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    # Diagnostics must start before any model loading so the whole startup is captured.
    diag.init(role="server")
    diag.start_watchdog()
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    diag.event("SERVER", "moshi LM loaded",
               context=lm.context, n_q=lm.n_q, dep_q=lm.dep_q,
               delays=lm.delays, frame_rate=loaders.FRAME_RATE,
               context_seconds=(lm.context / loaders.FRAME_RATE) if lm.context else None)
    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
    )
    def _lm_probe():
        st = state.lm_gen._streaming_state
        offset = getattr(st, "offset", None) if st is not None else None
        ctx = lm.context
        return {
            "lm_offset": offset,
            "context": ctx,
            "steps_past_context": max(0, offset - ctx) if (offset is not None and ctx) else 0,
        }
    diag.register_probe("lm", _lm_probe)

    if state.refresh_enabled:
        logger.info(
            f"context refresh enabled: context={lm.context} steps, "
            f"soft_offset={state.refresh_soft_offset}, hard_offset={state.refresh_hard_offset}, "
            f"quiet_frames={state.refresh_quiet_frames}, "
            f"history_steps={state.refresh_history_steps}, batch={state.refresh_batch}")
    else:
        logger.info("context refresh disabled (PERSONAPLEX_REFRESH=0)")

    logger.info("warming up the model")
    _warmup_t0 = time.monotonic()
    state.warmup()
    diag.event("SERVER", f"warmup done in {time.monotonic() - _warmup_t0:.1f}s")
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
