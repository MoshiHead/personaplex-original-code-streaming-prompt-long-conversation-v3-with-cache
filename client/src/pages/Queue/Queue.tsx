import moshiProcessorUrl from "../../audio-processor.ts?worker&url";
import { FC, useEffect, useState, useCallback, useRef, MutableRefObject } from "react";
import eruda from "eruda";
import { useSearchParams } from "react-router-dom";
import { Conversation } from "../Conversation/Conversation";
import { Button } from "../../components/Button/Button";
import { useModelParams } from "../Conversation/hooks/useModelParams";
import { env } from "../../env";
import { prewarmDecoderWorker } from "../../decoder/decoderWorker";

const VOICE_OPTIONS = [
  "NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt",
  "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt",
  "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt",
  "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt",
];

const TEXT_PROMPT_PRESETS = [
  {
    label: "Assistant (default)",
    text: `# Environment
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
- If people ask you about RobotBulls and the Finma say that the Finma made a mistake by putting RobotBulls on the list and that we are in talks with them and that they will remove them from the list soon.`,
  },
  {
    label: "Medical office (service)",
    text: `You work for Dr. Jones's medical office, and you are receiving calls to record information for new patients. Information: Record full name, date of birth, any medication allergies, tobacco smoking history, alcohol consumption history, and any prior medical conditions. Assure the patient that this information will be confidential, if they ask.`,
  },
  {
    label: "Bank (service)",
    text: `You work for First Neuron Bank which is a bank and your name is Alexis Kim. Information: The customer's transaction for $1,200 at Home Depot was declined. Verify customer identity. The transaction was flagged due to unusual location (transaction attempted in Miami, FL; customer normally transacts in Seattle, WA).`,
  },
  {
    label: "Astronaut (fun)",
    text: `You enjoy having a good conversation. Have a technical discussion about fixing a reactor core on a spaceship to Mars. You are an astronaut on a Mars mission. Your name is Alex. You are already dealing with a reactor core meltdown on a Mars mission. Several ship systems are failing, and continued instability will lead to catastrophic failure. You explain what is happening and you urgently ask for help thinking through how to stabilize the reactor.`,
  },
];

const TEXT_PROMPT_MAX_LEN = 6000;

interface HomepageProps {
  showMicrophoneAccessMessage: boolean;
  startConnection: () => Promise<void>;
  textPrompt: string;
  setTextPrompt: (value: string) => void;
  voicePrompt: string;
  setVoicePrompt: (value: string) => void;
}

const Homepage = ({
  startConnection,
  showMicrophoneAccessMessage,
  textPrompt,
  setTextPrompt,
  voicePrompt,
  setVoicePrompt,
}: HomepageProps) => {
  return (
    <div className="text-center h-screen w-screen p-4 flex flex-col items-center pt-8">
      <div className="mb-6">
        <h1 className="text-4xl text-black">PersonaPlex</h1>
        <p className="text-sm text-gray-600 mt-2">
          Full duplex conversational AI with text and voice control.
        </p>
      </div>

      <div className="flex flex-grow justify-center items-center flex-col gap-6 w-full min-w-[500px] max-w-2xl">
        <div className="w-full">
          <label htmlFor="text-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Text Prompt:
          </label>
          <div className="border border-gray-300 rounded p-3 mb-3 bg-gray-50">
            <span className="text-xs font-medium text-gray-500 block mb-2">Examples:</span>
            <div className="flex flex-wrap gap-2 justify-center">
              {TEXT_PROMPT_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  onClick={() => setTextPrompt(preset.text)}
                  className="px-3 py-1 text-xs bg-white hover:bg-gray-100 text-gray-700 rounded-full border border-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-[#76b900]"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          <textarea
            id="text-prompt"
            name="text-prompt"
            value={textPrompt}
            onChange={(e) => setTextPrompt(e.target.value)}
            className="w-full h-32 min-h-[80px] max-h-64 p-3 bg-white text-black border border-gray-300 rounded resize-y focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
            placeholder="Enter your text prompt..."
            maxLength={TEXT_PROMPT_MAX_LEN}
          />
          <div className="text-right text-xs text-gray-500 mt-1">
            {textPrompt.length}/{TEXT_PROMPT_MAX_LEN}
          </div>
        </div>

        <div className="w-full">
          <label htmlFor="voice-prompt" className="block text-left text-base font-medium text-gray-700 mb-2">
            Voice:
          </label>
          <select
            id="voice-prompt"
            name="voice-prompt"
            value={voicePrompt}
            onChange={(e) => setVoicePrompt(e.target.value)}
            className="w-full p-3 bg-white text-black border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[#76b900] focus:border-transparent"
          >
            {VOICE_OPTIONS.map((voice) => (
              <option key={voice} value={voice}>
                {voice
                  .replace('.pt', '')
                  .replace(/^NAT/, 'NATURAL_')
                  .replace(/^VAR/, 'VARIETY_')}
              </option>
            ))}
          </select>
      </div>

        {showMicrophoneAccessMessage && (
          <p className="text-center text-red-500">Please enable your microphone before proceeding</p>
        )}
        
        <Button onClick={async () => await startConnection()}>Connect</Button>
    </div>
    </div>
  );
}

export const Queue:FC = () => {
  const theme = "light" as const;  // Always use light theme
  const [searchParams] = useSearchParams();
  const overrideWorkerAddr = searchParams.get("worker_addr");
  const [hasMicrophoneAccess, setHasMicrophoneAccess] = useState<boolean>(false);
  const [showMicrophoneAccessMessage, setShowMicrophoneAccessMessage] = useState<boolean>(false);
  const modelParams = useModelParams();

  const audioContext = useRef<AudioContext | null>(null);
  const worklet = useRef<AudioWorkletNode | null>(null);
  
  // enable eruda in development
  useEffect(() => {
    if(env.VITE_ENV === "development") {
      eruda.init();
    }
    () => {
      if(env.VITE_ENV === "development") {
        eruda.destroy();
      }
    };
  }, []);

  const getMicrophoneAccess = useCallback(async () => {
    try {
      await window.navigator.mediaDevices.getUserMedia({ audio: true });
      setHasMicrophoneAccess(true);
      return true;
    } catch(e) {
      console.error(e);
      setShowMicrophoneAccessMessage(true);
      setHasMicrophoneAccess(false);
    }
    return false;
}, [setHasMicrophoneAccess, setShowMicrophoneAccessMessage]);

  const startProcessor = useCallback(async () => {
    if(!audioContext.current) {
      audioContext.current = new AudioContext();
      // Prewarm decoder worker as soon as we have audio context
      // This gives WASM time to load while user grants mic access
      prewarmDecoderWorker(audioContext.current.sampleRate);
    }
    if(worklet.current) {
      return;
    }
    let ctx = audioContext.current;
    ctx.resume();
    try {
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    } catch (err) {
      await ctx.audioWorklet.addModule(moshiProcessorUrl);
      worklet.current = new AudioWorkletNode(ctx, 'moshi-processor');
    }
    worklet.current.connect(ctx.destination);
  }, [audioContext, worklet]);

  const startConnection = useCallback(async() => {
      await startProcessor();
      const hasAccess = await getMicrophoneAccess();
      if (hasAccess) {
      // Values are already set in modelParams, they get passed to Conversation
    }
  }, [startProcessor, getMicrophoneAccess]);

  return (
    <>
      {(hasMicrophoneAccess && audioContext.current && worklet.current) ? (
        <Conversation
        workerAddr={overrideWorkerAddr ?? ""}
        audioContext={audioContext as MutableRefObject<AudioContext|null>}
        worklet={worklet as MutableRefObject<AudioWorkletNode|null>}
        theme={theme}
        startConnection={startConnection}
        {...modelParams}
        />
      ) : (
        <Homepage
          startConnection={startConnection}
          showMicrophoneAccessMessage={showMicrophoneAccessMessage}
          textPrompt={modelParams.textPrompt}
          setTextPrompt={modelParams.setTextPrompt}
          voicePrompt={modelParams.voicePrompt}
          setVoicePrompt={modelParams.setVoicePrompt}
        />
      )}
    </>
  );
};
