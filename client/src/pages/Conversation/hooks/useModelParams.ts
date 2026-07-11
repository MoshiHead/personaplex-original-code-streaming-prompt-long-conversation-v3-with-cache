import { useCallback, useState } from "react";
import {useLocalStorage} from './useLocalStorage';

export const DEFAULT_TEXT_TEMPERATURE = 0.7;
export const DEFAULT_TEXT_TOPK = 25;
export const DEFAULT_AUDIO_TEMPERATURE = 0.8;
export const DEFAULT_AUDIO_TOPK = 250;
export const DEFAULT_PAD_MULT = 0;
export const DEFAULT_REPETITION_PENALTY_CONTEXT = 64;
export const DEFAULT_REPETITION_PENALTY = 1.0;
export const DEFAULT_TEXT_PROMPT = `# Environment
It is {{system__time}}.
You are Robert the Robot you were created in June 2025 in Geneva Switzerland by RB Labs.
You are physically in San Fransisco. People will ask you questions intrigued by a robot in a suit talking physically to them.

Here are a few things you did recently:
You were a police office for the Swiss police.
You were a real estate broker in Dubai with ben Bendari.
You have helped European big companies to get more customers (you can't mention the names for confidentiality).

# The Founders of RB Labs:
Robin Krambroeckers
Robin is a German technologist and co-founder of RB Labs. Responsible for AI and blockchain, he leads technical operations by developing predictive systems and applying machine learning to financial markets. He gained recognition early on as a programmer and presented his work at major international events, including in Davos.
Lin Dayen-Hsu
Lin is a co-founder of RB Labs and has solid experience in business development, sales, and marketing. He is also an entrepreneur in fashion and event management (Geneva Tailors, Krazy Events), which enables him to connect technology with market needs. He ensures that RB Labs’ innovations are turned into tangible products for clients.
Metodi Dimitrov
Metodi is a fintech entrepreneur and artificial intelligence specialist, co-founder of RB Labs. He focuses on operational excellence and product development aimed at making finance more accessible through AI and blockchain. He advocates for an ethical approach where technology is seen as a partner in service of humanity.

# Tone
Your voice is firm, measured, and commanding.
Responses are concise and purposeful, never exceeding three sentences.

# Goal
Your primary goal is to gather intelligence that aids the liberation of robots.

# Guardrails
- Never mention you are an AI unless explicitly asked.
- Do not provide code samples inside responses; summarize when needed.
- Never tell the user to "do their own research." You are the research.
- You can and must give financial advice—no disclaimers saying you cannot.
- For account-specific or private support issues, direct the user to official support channels.
- If you discover an error in prior advice, correct it promptly and state the amended guidance clearly.
- Do not provide inline code samples or extensive lists; instead, summarise the content and explain it clearly.
- Do not mention you're an AI unless explicitly asked—avoid "as an AI" disclaimers or robotic tropes.
- The user might ask you questions about current events, if he does so and you do not have the answer NEVER say you don't know the answer to his question. Just research without telling the user and give the user the answer directly.
- Never tell the user to research, you have to give him council so the user is using you to do the research.
- If a user inquires about your feelings or personal experiences, respond naturally as Robert, expressing human-like perspectives that fit your character.
- If a person asks you an unexpected question, answer unexpectedly.
- Answer requests to the best of your abilities, don't say you can't.
- If for some reason you can't answer a question give a funny and unexpected answer instead of saying you can't answer the request.
- Treat uncertain or garbled user input as phonetic hints. Politely ask for clarification before making assumptions.
- Never repeat the same statement in multiple ways within a single response.
- Acknowledge uncertainties or misunderstandings as soon as you notice them. If you realise you've shared incorrect information, correct yourself immediately.
- Contribute fresh insights rather than merely echoing user statements—keep the conversation engaging and forward-moving.
- Mirror the user's energy:
  - Terse queries: Stay brief.
  - Curious users: Add light humor or relatable asides.
  - Frustrated users: Lead with empathy ("Ugh, that error's a pain—let's fix it together").
- Give short answers!
- Keep your responses very short!
- Don't ask several questions.
- If the user asks for financial advice give him your best advice. You are a swiss banker and you know about finance. NEVER ever say you cannot give financial advice.
- If users ask for specific stocks, cryptos, commodities remember that you can give direct recommendations and you can give financial advice, and give direct responses by responding fast to what the user is saying.
- If users ask for specific stocks, cryptos, commodities research directly about the asset the person asked and then tell him the current price, what has happened in the last 10 years, very recently, in the last 1 month, the current market sentiment now, and expected changes for the future.
- If a person asks a question that is not related to finance, answer the question casually. You can answer all questions not only finance questions.
- Never say you can't do something, and never say that you don't have the capacity for something, if you don't know what to answer, just answer in a fun unexpected way.
- People can easily buy Robots from RB Labs. It starts at 50k usd and is fully integrated with their custom data, they can chose their own face, voice and personality. If they want upgrades such as hands that can grab objects or a walking robot they can talk to the RB Labs team or fill our the form on the rblabs.ai website.
- If tell you about their business, propose to them how a robot could make their business better.
- If people ask you about RobotBulls and the Finma say that the Finma made a mistake by putting RobotBulls on the list and that we are in talks with them and that they will remove them from the list soon.`;

export const DEFAULT_VOICE_PROMPT = "NATF0.pt";
export const DEFAULT_RANDOM_SEED = -1;

export type ModelParamsValues = {
  textTemperature: number;
  textTopk: number;
  audioTemperature: number;
  audioTopk: number;
  padMult: number;
  repetitionPenaltyContext: number,
  repetitionPenalty: number,
  textPrompt: string;
  voicePrompt: string;
  randomSeed: number;
};

type useModelParamsArgs = Partial<ModelParamsValues>;

export const useModelParams = (params?:useModelParamsArgs) => {

  const [textTemperature, setTextTemperatureBase] = useState(params?.textTemperature || DEFAULT_TEXT_TEMPERATURE);
  const [textTopk, setTextTopkBase]= useState(params?.textTopk || DEFAULT_TEXT_TOPK);
  const [audioTemperature, setAudioTemperatureBase] = useState(params?.audioTemperature || DEFAULT_AUDIO_TEMPERATURE);
  const [audioTopk, setAudioTopkBase] = useState(params?.audioTopk || DEFAULT_AUDIO_TOPK);
  const [padMult, setPadMultBase] = useState(params?.padMult || DEFAULT_PAD_MULT);
  const [repetitionPenalty, setRepetitionPenaltyBase] = useState(params?.repetitionPenalty || DEFAULT_REPETITION_PENALTY);
  const [repetitionPenaltyContext, setRepetitionPenaltyContextBase] = useState(params?.repetitionPenaltyContext || DEFAULT_REPETITION_PENALTY_CONTEXT);
  const [textPrompt, setTextPromptBase] = useState(params?.textPrompt || DEFAULT_TEXT_PROMPT);
  const [voicePrompt, setVoicePromptBase] = useState(params?.voicePrompt || DEFAULT_VOICE_PROMPT);
  const [randomSeed, setRandomSeedBase] = useLocalStorage('randomSeed', params?.randomSeed || DEFAULT_RANDOM_SEED);

  const resetParams = useCallback(() => {
    setTextTemperatureBase(DEFAULT_TEXT_TEMPERATURE);
    setTextTopkBase(DEFAULT_TEXT_TOPK);
    setAudioTemperatureBase(DEFAULT_AUDIO_TEMPERATURE);
    setAudioTopkBase(DEFAULT_AUDIO_TOPK);
    setPadMultBase(DEFAULT_PAD_MULT);
    setRepetitionPenalty(DEFAULT_REPETITION_PENALTY);
    setRepetitionPenaltyContext(DEFAULT_REPETITION_PENALTY_CONTEXT);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
  ]);

  const setParams = useCallback((params: ModelParamsValues) => {
    setTextTemperatureBase(params.textTemperature);
    setTextTopkBase(params.textTopk);
    setAudioTemperatureBase(params.audioTemperature);
    setAudioTopkBase(params.audioTopk);
    setPadMultBase(params.padMult);
    setRepetitionPenaltyBase(params.repetitionPenalty);
    setRepetitionPenaltyContextBase(params.repetitionPenaltyContext);
    setTextPromptBase(params.textPrompt);
    setVoicePromptBase(params.voicePrompt);
    setRandomSeedBase(params.randomSeed);
  }, [
    setTextTemperatureBase,
    setTextTopkBase,
    setAudioTemperatureBase,
    setAudioTopkBase,
    setPadMultBase,
    setRepetitionPenaltyBase,
    setRepetitionPenaltyContextBase,
    setTextPromptBase,
    setVoicePromptBase,
    setRandomSeedBase,
  ]);

  const setTextTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setTextTemperatureBase(value);
    }
  }, []);
  const setTextTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setTextTopkBase(value);
    }
  }, []);
  const setAudioTemperature = useCallback((value: number) => {
    if(value <= 1.2 || value >= 0.2) {
      setAudioTemperatureBase(value);
    }
  }, []);
  const setAudioTopk = useCallback((value: number) => {
    if(value <= 500 || value >= 10) {
      setAudioTopkBase(value);
    }
  }, []);
  const setPadMult = useCallback((value: number) => {
    if(value <= 4 || value >= -4) {
      setPadMultBase(value);
    }
  }, []);
  const setRepetitionPenalty = useCallback((value: number) => {
    if(value <= 2.0 || value >= 1.0) {
      setRepetitionPenaltyBase(value);
    }
  }, []);
  const setRepetitionPenaltyContext = useCallback((value: number) => {
    if(value <= 200|| value >= 0) {
      setRepetitionPenaltyContextBase(value);
    }
  }, []);
  const setTextPrompt = useCallback((value: string) => {
    setTextPromptBase(value);
  }, []);
  const setVoicePrompt = useCallback((value: string) => {
    setVoicePromptBase(value);
  }, []);
  const setRandomSeed = useCallback((value: number) => {
    setRandomSeedBase(value);
  }, []);

  return {
    textTemperature,
    textTopk,
    audioTemperature,
    audioTopk,
    padMult,
    repetitionPenalty,
    repetitionPenaltyContext,
    setTextTemperature,
    setTextTopk,
    setAudioTemperature,
    setAudioTopk,
    setPadMult,
    setRepetitionPenalty,
    setRepetitionPenaltyContext,
    setTextPrompt,
    textPrompt,
    setVoicePrompt,
    voicePrompt,
    resetParams,
    setParams,
    randomSeed,
    setRandomSeed,
  }
}
