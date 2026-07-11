# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Deep diagnostics / instrumentation for the PersonaPlex streaming stack.

This module is *pure observability*: it never changes model behavior, prompts,
sampling parameters, or conversation logic. It provides:

  - structured, rotating log files (debug + error) with per-conversation
    elapsed timestamps and component tags;
  - heartbeats: every streaming component reports "I made progress" and a
    background watchdog thread flags any component that stops progressing;
  - counters/gauges for frames, tokens, bytes, latencies;
  - a conversation state machine (user speaking / model speaking / idle)
    derived from input/output audio energy, with logged transitions;
  - MODEL_STREAM summaries: LM step offset vs. context window capacity,
    KV-cache write position, raw generated tokens, step latency, GPU memory;
  - freeze snapshots: when a component stalls, all Python thread stacks and
    asyncio task stacks are dumped to logs/diagnostics/freeze_snapshot_*.txt
    without crashing or pausing the application.

Everything is controlled by environment variables (all optional):

  PERSONAPLEX_DEBUG              "1" (default) / "0"  master switch
  PERSONAPLEX_LOG_DIR            log directory (default: ./logs)
  PERSONAPLEX_WATCHDOG_INTERVAL  seconds between HEALTH reports (default 5)
  PERSONAPLEX_STALL_TIMEOUT      default staleness before a component is
                                 declared stuck and a snapshot is taken (default 15)
  PERSONAPLEX_STEP_LOG_EVERY     frames between MODEL_STREAM summaries (default 25 = 2 s)
  PERSONAPLEX_FRAME_LOG          "1" (default) / "0"  per-frame DEBUG lines (file only)

The module-level singleton is `diag`. Until `diag.init()` is called (done by
moshi.server at startup, before model loading), `diag.enabled` is False and
every hook is a cheap no-op, so importing this module from library code
(e.g. moshi.offline) costs nothing.

Design notes for correctness of the numbers reported here:

  - The authoritative LM step counter is `LMGen._streaming_state.offset`,
    a plain Python int incremented in `LMGen.step`. The per-layer
    `offset_cpu` ints inside the transformer are NOT reliable once CUDA
    graphs are captured (the Python forward body no longer runs on replay),
    so we never report them.
  - The KV cache `end_offset` tensor DOES advance inside graph replays.
    Reading it requires a device sync, so it is only read on the inference
    thread (in `model_stream_summary`), never from the watchdog thread.
  - The watchdog thread only reads Python ints/floats and CUDA memory
    statistics (which do not synchronize the device).
"""

import asyncio
import datetime
import faulthandler
import io
import logging
import logging.handlers
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover - torch is always present in practice
    torch = None  # type: ignore


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Text stream special tokens (see moshi/models/lm.py and server.py):
# 0 = EPAD (end of padding), 3 = PAD. Anything else is a word piece
# (or BOS=1 / EOS=2, which the server also treats as sendable text).
_TEXT_EPAD = 0
_TEXT_PAD = 3

# RMS thresholds used only to *label* conversation states in the logs.
# 80 ms frames; hysteresis avoids flapping on breaths/noise.
_IN_SPEECH_RMS = 0.015
_OUT_SPEECH_RMS = 0.010
_HYSTERESIS_FRAMES = 4  # 4 frames = 320 ms


class Diagnostics:
    """Process-wide diagnostics hub. Thread-safe for the access patterns used
    here: the asyncio/inference thread writes counters and heartbeats, the
    watchdog thread reads them."""

    def __init__(self):
        self.enabled = False
        self.log_dir = "logs"
        self.watchdog_interval = 5.0
        self.stall_timeout_default = 15.0
        self.step_log_every = 25
        self.frame_log = True

        self._lock = threading.RLock()
        self._logger: Optional[logging.Logger] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._fault_file = None

        # name -> (monotonic_time, fields dict)
        self.heartbeats: Dict[str, Any] = {}
        # component name -> staleness timeout in seconds (armed during a conversation)
        self.stall_timeouts: Dict[str, float] = {}
        self._last_stall_warn: Dict[str, float] = {}
        self.counters: Dict[str, float] = {}
        self._counters_prev: Dict[str, float] = {}
        self.gauges: Dict[str, Any] = {}
        self.probes: Dict[str, Callable[[], dict]] = {}

        self._snapshot_seq = 0
        self._conv_seq = 0
        self._stalls_suspended_until = 0.0
        self.conv_t0: Optional[float] = None  # monotonic; None = no active conversation
        self.conv_active = False

        # Conversation state machine
        self.conv_state = "idle"
        self._user_speaking = False
        self._model_speaking = False
        self._in_run = 0   # consecutive frames agreeing with a flip of user_speaking
        self._out_run = 0
        self.user_turns = 0
        self.model_turns = 0

        # Text stream tracking
        self._pad_run = 0            # consecutive PAD/EPAD text tokens
        self._last_word_mono: Optional[float] = None

        # LM step tracking
        self._last_lm_offset = -1
        self._step_ms_accum = 0.0
        self._step_ms_max = 0.0
        self._step_accum_n = 0

    # ------------------------------------------------------------------ setup

    def init(self, role: str = "server"):
        """Initialize logging. Idempotent. Must be called before model loading
        so that every later event is captured."""
        if self._logger is not None:
            return
        self.enabled = _env_flag("PERSONAPLEX_DEBUG", True)
        if not self.enabled:
            return
        self.log_dir = os.environ.get("PERSONAPLEX_LOG_DIR", "logs")
        self.watchdog_interval = _env_float("PERSONAPLEX_WATCHDOG_INTERVAL", 5.0)
        self.stall_timeout_default = _env_float("PERSONAPLEX_STALL_TIMEOUT", 15.0)
        self.step_log_every = max(1, _env_int("PERSONAPLEX_STEP_LOG_EVERY", 25))
        self.frame_log = _env_flag("PERSONAPLEX_FRAME_LOG", True)

        snap_dir = os.path.join(self.log_dir, "diagnostics")
        os.makedirs(snap_dir, exist_ok=True)

        logger = logging.getLogger("personaplex.diag")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", datefmt="%H:%M:%S")

        debug_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, "personaplex_debug.log"),
            maxBytes=64 * 1024 * 1024, backupCount=5, encoding="utf-8")
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(fmt)
        logger.addHandler(debug_handler)

        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(self.log_dir, "personaplex_error.log"),
            maxBytes=16 * 1024 * 1024, backupCount=3, encoding="utf-8")
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(fmt)
        logger.addHandler(error_handler)

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        logger.addHandler(console)

        self._logger = logger

        # Low-level safety net: SIGSEGV/deadlock tracebacks go to a dedicated file.
        try:
            self._fault_file = open(os.path.join(self.log_dir, "personaplex_fault.log"), "w")
            faulthandler.enable(file=self._fault_file)
        except Exception:
            pass

        self.event("DIAG", f"diagnostics initialized role={role} pid={os.getpid()} "
                           f"log_dir={os.path.abspath(self.log_dir)} "
                           f"watchdog_interval={self.watchdog_interval}s "
                           f"stall_timeout={self.stall_timeout_default}s "
                           f"step_log_every={self.step_log_every}")

    def start_watchdog(self):
        if not self.enabled or self._watchdog_thread is not None:
            return
        t = threading.Thread(target=self._watchdog_run, name="personaplex-watchdog", daemon=True)
        t.start()
        self._watchdog_thread = t
        self.event("DIAG", "watchdog thread started")

    def register_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def loop_ticker(self):
        """Coroutine run inside the server event loop. Its heartbeat proves the
        event loop itself is not blocked; it also snapshots the asyncio task
        list from inside the loop (the only thread-safe place to do so).
        Never returns (it is cancelled with the session tasks)."""
        while True:
            try:
                if self.enabled:
                    self.beat("event_loop")
                    tasks = asyncio.all_tasks()
                    self.gauges["asyncio_tasks"] = len(tasks)
                    self.gauges["asyncio_task_names"] = ",".join(
                        sorted(t.get_name() for t in tasks))[:300]
            except Exception:
                pass
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------ primitives

    def _elapsed_str(self) -> str:
        if self.conv_t0 is None:
            return "--:--.-"
        e = time.monotonic() - self.conv_t0
        return f"{int(e // 60):02d}:{e % 60:06.3f}"

    def event(self, component: str, msg: str, level: str = "info", **fields):
        """Structured log line: wall clock (from formatter), conversation
        elapsed time, component tag, message, key=value fields."""
        if not self.enabled or self._logger is None:
            return
        if fields:
            kv = " ".join(f"{k}={v}" for k, v in fields.items())
            msg = f"{msg} | {kv}"
        line = f"[conv {self._elapsed_str()}] [{component}] {msg}"
        lvl = {"debug": logging.DEBUG, "info": logging.INFO,
               "warning": logging.WARNING, "error": logging.ERROR}.get(level, logging.INFO)
        self._logger.log(lvl, line)

    def beat(self, name: str, **fields):
        """Record that component `name` made progress just now."""
        if not self.enabled:
            return
        self.heartbeats[name] = (time.monotonic(), fields or None)

    def count(self, name: str, n: float = 1):
        if not self.enabled:
            return
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def gauge(self, name: str, value: Any):
        if not self.enabled:
            return
        self.gauges[name] = value

    def set_stall_timeout(self, name: str, seconds: Optional[float] = None):
        """Arm the watchdog for component `name`. If its heartbeat goes stale
        for longer than `seconds`, a warning is logged and a freeze snapshot
        is written."""
        if not self.enabled:
            return
        self.stall_timeouts[name] = seconds if seconds is not None else self.stall_timeout_default

    def suspend_stall_checks(self, seconds: float):
        """Disable stall detection for `seconds` — used around planned pauses
        (e.g. prompt re-injection) so the watchdog doesn't dump false-positive
        freeze snapshots. Calling again overwrites the deadline, so a short
        grace period after the pause re-arms detection cleanly."""
        if not self.enabled:
            return
        self._stalls_suspended_until = time.monotonic() + seconds

    def register_probe(self, name: str, fn: Callable[[], dict]):
        """Register a callable returning a dict of live state; invoked by the
        watchdog on every HEALTH report and by freeze snapshots. Must be cheap
        and must not touch CUDA tensors (watchdog runs on its own thread)."""
        if not self.enabled:
            return
        self.probes[name] = fn

    # ------------------------------------------------- conversation lifecycle

    def new_conversation(self, meta: Optional[dict] = None):
        if not self.enabled:
            return
        with self._lock:
            self._conv_seq += 1
            self.conv_t0 = time.monotonic()
            self.conv_active = True
            self.counters.clear()
            self._counters_prev.clear()
            self.heartbeats.clear()
            self._last_stall_warn.clear()
            self.conv_state = "idle"
            self._user_speaking = False
            self._model_speaking = False
            self._in_run = 0
            self._out_run = 0
            self.user_turns = 0
            self.model_turns = 0
            self._pad_run = 0
            self._last_word_mono = None
            self._step_ms_accum = 0.0
            self._step_ms_max = 0.0
            self._step_accum_n = 0
        started = datetime.datetime.now().isoformat(timespec="milliseconds")
        self.event("CONVERSATION", f"=== conversation #{self._conv_seq} started at {started} ===",
                   **(meta or {}))

    def end_conversation(self, reason: str = ""):
        if not self.enabled:
            return
        elapsed = self._elapsed_str()
        self.event("CONVERSATION",
                   f"=== conversation #{self._conv_seq} ended after {elapsed} ===",
                   reason=reason or "n/a",
                   user_turns=self.user_turns, model_turns=self.model_turns,
                   lm_steps=int(self.counters.get("lm_steps", 0)),
                   frames_in=int(self.counters.get("frames_in", 0)),
                   frames_out=int(self.counters.get("frames_out", 0)))
        with self._lock:
            self.conv_active = False
            # Disarm the watchdog so it does not fire between sessions.
            self.stall_timeouts.clear()

    def _set_conv_state(self, new_state: str):
        if new_state != self.conv_state:
            self.event("STATE", f"{self.conv_state.upper()} -> {new_state.upper()}",
                       user_turns=self.user_turns, model_turns=self.model_turns)
            self.conv_state = new_state

    # --------------------------------------------------------- audio pipeline

    def on_input_frame(self, rms: float):
        """One 80 ms user-audio frame entered the pipeline (pre-encode)."""
        if not self.enabled:
            return
        self.beat("audio_in_frame", rms=round(rms, 4))
        self.count("frames_in")
        self.gauges["in_rms"] = round(rms, 5)
        speaking = rms >= _IN_SPEECH_RMS
        if speaking != self._user_speaking:
            self._in_run += 1
            if self._in_run >= _HYSTERESIS_FRAMES:
                self._user_speaking = speaking
                self._in_run = 0
                if speaking:
                    self.user_turns += 1
                    self.event("TURN", "USER_STARTED_SPEAKING", rms=round(rms, 4))
                else:
                    self.event("TURN", "USER_AUDIO_COMPLETE (input energy dropped)")
                self._update_composite_state()
        else:
            self._in_run = 0

    def on_output_frame(self, out_rms: float, text_token: int,
                        encode_ms: float = 0.0, step_ms: float = 0.0,
                        decode_ms: float = 0.0, frame_wall_ms: float = 0.0):
        """One 80 ms model-audio frame left the decoder toward the client."""
        if not self.enabled:
            return
        self.beat("audio_out_frame", rms=round(out_rms, 4))
        self.count("frames_out")
        self.gauges["out_rms"] = round(out_rms, 5)
        self.gauges["last_frame_ms"] = round(frame_wall_ms, 1)

        # Text-channel bookkeeping: how long since the model produced a real word?
        if text_token in (_TEXT_PAD, _TEXT_EPAD):
            self._pad_run += 1
        else:
            self._pad_run = 0
            self._last_word_mono = time.monotonic()
            self.count("text_words")
        self.gauges["text_pad_run"] = self._pad_run

        speaking = out_rms >= _OUT_SPEECH_RMS
        if speaking != self._model_speaking:
            self._out_run += 1
            if self._out_run >= _HYSTERESIS_FRAMES:
                self._model_speaking = speaking
                self._out_run = 0
                if speaking:
                    self.model_turns += 1
                    self.event("TURN", "MODEL_SPEECH_STARTED", rms=round(out_rms, 4))
                else:
                    self.event("TURN", "MODEL_SPEECH_FINISHED (output energy dropped)")
                self._update_composite_state()
        else:
            self._out_run = 0

        if self.frame_log:
            self.event(
                "FRAME",
                f"n={int(self.counters.get('frames_out', 0))}",
                level="debug",
                in_rms=self.gauges.get("in_rms"),
                out_rms=round(out_rms, 5),
                text_tok=text_token,
                pad_run=self._pad_run,
                enc_ms=round(encode_ms, 1),
                step_ms=round(step_ms, 1),
                dec_ms=round(decode_ms, 1),
                wall_ms=round(frame_wall_ms, 1),
            )

    def _update_composite_state(self):
        if self._user_speaking and self._model_speaking:
            self._set_conv_state("both_speaking")
        elif self._user_speaking:
            self._set_conv_state("user_speaking")
        elif self._model_speaking:
            self._set_conv_state("model_speaking")
        else:
            self._set_conv_state("idle")

    def on_text_piece(self, token: int, piece: str):
        if not self.enabled:
            return
        self.beat("text_out")
        self.event("TEXT", f"token={token} piece={piece!r}", level="debug")

    # ------------------------------------------------------------- LM hooks

    def on_lm_step(self, dur_ms: float, offset: int, produced_output: bool):
        """Called from LMGen.step (inference thread) after every generation step."""
        if not self.enabled:
            return
        self.beat("lm_step", offset=offset)
        self.count("lm_steps")
        if produced_output:
            self.count("lm_steps_with_output")
            self.beat("lm_output", offset=offset)
        self._last_lm_offset = offset
        self.gauges["lm_offset"] = offset
        self.gauges["last_step_ms"] = round(dur_ms, 1)
        with self._lock:
            self._step_ms_accum += dur_ms
            self._step_accum_n += 1
            if dur_ms > self._step_ms_max:
                self._step_ms_max = dur_ms

    def model_stream_summary(self, lm_gen, tokens=None):
        """Periodic deep-dive into model internals. MUST be called on the
        inference thread: it reads the CUDA-side KV-cache offset (device sync)
        and copies the last generated tokens to CPU."""
        if not self.enabled:
            return
        try:
            st = lm_gen._streaming_state
            offset = st.offset if st is not None else -1
            ctx = getattr(lm_gen.lm_model, "context", None)
            past_ctx = max(0, offset - ctx) if ctx else 0
            kv_end = None
            try:
                mha_state = lm_gen.lm_model.transformer.layers[0].self_attn._streaming_state
                if mha_state is not None:
                    kv_end = int(mha_state.kv_cache.end_offset.item())
            except Exception:
                pass
            tok_list = None
            if tokens is not None:
                try:
                    tok_list = tokens[0, :, 0].detach().cpu().tolist()
                except Exception:
                    pass
            with self._lock:
                n = max(1, self._step_accum_n)
                step_avg = self._step_ms_accum / n
                step_max = self._step_ms_max
                self._step_ms_accum = 0.0
                self._step_ms_max = 0.0
                self._step_accum_n = 0
            gpu = ""
            if torch is not None and torch.cuda.is_available():
                gpu = (f" gpu_alloc_mb={torch.cuda.memory_allocated() // (1 << 20)}"
                       f" gpu_reserved_mb={torch.cuda.memory_reserved() // (1 << 20)}")
            secs_since_word = None
            if self._last_word_mono is not None:
                secs_since_word = round(time.monotonic() - self._last_word_mono, 1)
            self.event(
                "MODEL_STREAM",
                f"lm_offset={offset} context={ctx} steps_past_context={past_ctx} "
                f"kv_end_offset={kv_end} "
                f"steps={int(self.counters.get('lm_steps', 0))} "
                f"frames_out={int(self.counters.get('frames_out', 0))} "
                f"step_ms_avg={step_avg:.1f} step_ms_max={step_max:.1f} "
                f"in_rms={self.gauges.get('in_rms')} out_rms={self.gauges.get('out_rms')} "
                f"text_pad_run={self._pad_run} secs_since_last_word={secs_since_word} "
                f"tokens={tok_list}{gpu}"
            )
        except Exception:
            self.event("MODEL_STREAM", f"summary failed:\n{traceback.format_exc()}",
                       level="warning")

    # -------------------------------------------------------------- watchdog

    def _gpu_stats(self) -> str:
        if torch is None or not torch.cuda.is_available():
            return "gpu=n/a"
        try:
            alloc = torch.cuda.memory_allocated() // (1 << 20)
            reserved = torch.cuda.memory_reserved() // (1 << 20)
            util = ""
            try:
                util = f" util={torch.cuda.utilization()}%"
            except Exception:
                pass
            return f"gpu: alloc={alloc}MB reserved={reserved}MB{util}"
        except Exception:
            return "gpu=err"

    def _health_report(self) -> str:
        now = time.monotonic()
        lines = [f"PERSONAPLEX HEALTH conv={self._elapsed_str()} state={self.conv_state} "
                 f"turns(user/model)={self.user_turns}/{self.model_turns} "
                 f"active={self.conv_active}"]
        # .copy() is atomic at the C level; plain iteration could race with
        # beat() calls from the inference thread.
        hb = sorted(self.heartbeats.copy().items())
        if hb:
            ages = " ".join(f"{name}={now - t:.1f}s" for name, (t, _f) in hb)
            lines.append(f"  last_progress: {ages}")
        with self._lock:
            items = sorted(self.counters.items())
            deltas = {k: v - self._counters_prev.get(k, 0) for k, v in items}
            self._counters_prev = dict(self.counters)
        if items:
            cnts = " ".join(f"{k}={int(v)}(+{int(deltas[k])})" for k, v in items)
            lines.append(f"  counters: {cnts}")
        if self.gauges:
            gs = " ".join(f"{k}={v}" for k, v in sorted(self.gauges.items())
                          if k != "asyncio_task_names")
            lines.append(f"  gauges: {gs}")
        for name, fn in list(self.probes.items()):
            try:
                vals = fn()
                lines.append(f"  probe[{name}]: " + " ".join(f"{k}={v}" for k, v in vals.items()))
            except Exception as exc:
                lines.append(f"  probe[{name}]: failed ({exc!r})")
        lines.append(f"  {self._gpu_stats()}")
        lines.append(f"  asyncio: tasks={self.gauges.get('asyncio_tasks', '?')} "
                     f"[{self.gauges.get('asyncio_task_names', '?')}]")
        return "\n".join(lines)

    def _check_stalls(self, now: float):
        if now < self._stalls_suspended_until:
            return
        for name, timeout in self.stall_timeouts.copy().items():
            entry = self.heartbeats.get(name)
            if entry is None:
                continue  # component never started; lifecycle events cover that
            age = now - entry[0]
            if age <= timeout:
                continue
            last_warn = self._last_stall_warn.get(name, 0.0)
            if now - last_warn < 60.0:
                continue  # rate-limit repeat snapshots for the same component
            self._last_stall_warn[name] = now
            hint = {
                "recv_loop": "WebSocket receive loop (client audio ingress) stopped",
                "audio_in_frame": "no decoded user PCM frames reaching the model "
                                  "(client stopped sending, or opus_reader starved)",
                "opus_loop": "main inference loop stopped iterating "
                             "(event loop blocked or coroutine stuck)",
                "lm_step": "LMGen.step() no longer being called or not returning "
                           "(model inference frozen or input starvation)",
                "lm_output": "LMGen.step() runs but produced no output tokens",
                "audio_out_frame": "mimi decoder stopped producing waveform frames",
                "ws_audio_out": "opus-encoded audio no longer sent on the WebSocket",
                "send_loop": "WebSocket send loop stopped iterating",
                "event_loop": "the asyncio event loop itself is blocked "
                              "(synchronous call stuck — see thread stacks)",
                "text_out": "text token stream stopped",
            }.get(name, "component stopped progressing")
            self.event("WATCHDOG",
                       f"WARNING: '{name}' made no progress for {age:.1f}s "
                       f"(timeout {timeout:.0f}s). Possible stuck component: {hint}. "
                       f"Dumping stack traces...",
                       level="warning")
            self.freeze_snapshot(f"stall of '{name}' ({age:.1f}s stale): {hint}")

    def _watchdog_run(self):
        while True:
            time.sleep(self.watchdog_interval)
            if not self.enabled:
                continue
            try:
                now = time.monotonic()
                if self.conv_active:
                    self.event("HEALTH", "\n" + self._health_report())
                    self._check_stalls(now)
            except Exception:
                try:
                    self.event("WATCHDOG", f"watchdog iteration failed:\n{traceback.format_exc()}",
                               level="warning")
                except Exception:
                    pass

    # ------------------------------------------------------- freeze snapshot

    def freeze_snapshot(self, reason: str) -> Optional[str]:
        """Dump every Python thread stack, asyncio task stack, diagnostics
        state and CUDA memory info to a file. Never raises."""
        if not self.enabled:
            return None
        try:
            self._snapshot_seq += 1
            conv_tag = self._elapsed_str().replace(":", "").split(".")[0]
            path = os.path.join(
                self.log_dir, "diagnostics",
                f"freeze_snapshot_{conv_tag}_{self._snapshot_seq:03d}.txt")
            buf = io.StringIO()
            buf.write("PERSONAPLEX FREEZE SNAPSHOT\n")
            buf.write(f"wall_time: {datetime.datetime.now().isoformat(timespec='milliseconds')}\n")
            buf.write(f"conversation_elapsed: {self._elapsed_str()}\n")
            buf.write(f"reason: {reason}\n\n")

            buf.write("=== HEALTH ===\n")
            buf.write(self._health_report())
            buf.write("\n\n=== PYTHON THREAD STACKS ===\n")
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                buf.write(f"\n--- thread {names.get(ident, '?')} (ident={ident}) ---\n")
                buf.write("".join(traceback.format_stack(frame)))

            buf.write("\n=== ASYNCIO TASKS ===\n")
            loop = self._loop
            if loop is not None:
                try:
                    tasks = asyncio.all_tasks(loop)
                    for task in tasks:
                        buf.write(f"\n--- task {task.get_name()} done={task.done()} ---\n")
                        buf.write(f"coro: {task.get_coro()!r}\n")
                        try:
                            task.print_stack(file=buf)
                        except Exception as exc:
                            buf.write(f"(could not print stack: {exc!r})\n")
                except Exception as exc:
                    buf.write(f"(could not enumerate tasks: {exc!r})\n")
            else:
                buf.write("(no event loop registered)\n")

            if torch is not None and torch.cuda.is_available():
                buf.write("\n=== CUDA MEMORY SUMMARY ===\n")
                try:
                    buf.write(torch.cuda.memory_summary())
                except Exception as exc:
                    buf.write(f"(memory_summary failed: {exc!r})\n")

            with open(path, "w", encoding="utf-8") as f:
                f.write(buf.getvalue())
            self.event("WATCHDOG", f"freeze snapshot written to {path}", level="warning")
            return path
        except Exception:
            try:
                self.event("WATCHDOG", f"freeze snapshot failed:\n{traceback.format_exc()}",
                           level="error")
            except Exception:
                pass
            return None


# Module-level singleton. `diag.enabled` stays False until `diag.init()` runs.
diag = Diagnostics()
