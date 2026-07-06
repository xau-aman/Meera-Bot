/**
 * Meera Voice Handler
 * Manages VC sessions: listen → STT → AI → TTS → speak
 * Trigger: "Hey Meera" activates, then natural conversation until silence timeout
 */

const {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  VoiceConnectionStatus,
  EndBehaviorType,
} = require("@discordjs/voice");
const { pipeline } = require("stream/promises");
const { createWriteStream, unlinkSync, existsSync, mkdirSync } = require("fs");
const { join } = require("path");
const { execSync } = require("child_process");
const { chat } = require("../lib/ai");
const prism = require("prism-media");

// Active voice sessions per guild
const sessions = new Map();

// Temp directory for audio files
const TEMP_DIR = join(__dirname, "..", "..", "temp");
if (!existsSync(TEMP_DIR)) mkdirSync(TEMP_DIR, { recursive: true });

// Session timeout (30s of no speech = end conversation)
const SILENCE_TIMEOUT = 30000;
// Wake word detection
const WAKE_WORDS = ["hey meera", "hi meera", "meera"];

class VoiceSession {
  constructor(connection, channel, guild) {
    this.connection = connection;
    this.channel = channel;
    this.guild = guild;
    this.player = createAudioPlayer();
    this.isActive = false; // becomes true after wake word
    this.isListening = false;
    this.isSpeaking = false;
    this.silenceTimer = null;
    this.activeUser = null; // who triggered the session
    this.listeners = new Map(); // userId -> audio stream

    connection.subscribe(this.player);

    this.player.on(AudioPlayerStatus.Idle, () => {
      this.isSpeaking = false;
      // Resume listening after speaking
      if (this.isActive) this.startListening();
    });

    this.player.on("error", (err) => {
      console.error("Audio player error:", err);
      this.isSpeaking = false;
    });
  }

  startListening() {
    if (this.isListening) return;
    this.isListening = true;
    this.resetSilenceTimer();

    const receiver = this.connection.receiver;

    receiver.speaking.on("start", (userId) => {
      if (this.isSpeaking) return; // don't listen while speaking

      // If not active yet, listen to everyone for wake word
      // If active, only listen to the active user (or anyone in convo)
      this.listenToUser(userId);
    });
  }

  listenToUser(userId) {
    if (this.listeners.has(userId)) return;

    const receiver = this.connection.receiver;
    const audioStream = receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 1500 },
    });

    const pcmStream = new prism.opus.Decoder({ rate: 48000, channels: 1, frameSize: 960 });
    const filePath = join(TEMP_DIR, `${userId}_${Date.now()}.pcm`);
    const writeStream = createWriteStream(filePath);

    audioStream.pipe(pcmStream).pipe(writeStream);

    writeStream.on("finish", async () => {
      this.listeners.delete(userId);
      await this.processAudio(userId, filePath);
    });

    this.listeners.set(userId, { audioStream, filePath });
  }

  async processAudio(userId, pcmPath) {
    try {
      // Convert PCM to WAV for Whisper
      const wavPath = pcmPath.replace(".pcm", ".wav");
      execSync(
        `ffmpeg -y -f s16le -ar 48000 -ac 1 -i "${pcmPath}" -ar 16000 -ac 1 "${wavPath}" 2>/dev/null`
      );

      // STT via Groq Whisper
      const text = await this.speechToText(wavPath);

      // Cleanup temp files
      this.cleanupFile(pcmPath);
      this.cleanupFile(wavPath);

      if (!text || text.trim().length < 2) return;

      console.log(`[Voice] ${userId}: "${text}"`);

      // Check for wake word if not active
      if (!this.isActive) {
        const lower = text.toLowerCase();
        const hasWakeWord = WAKE_WORDS.some((w) => lower.includes(w));
        if (!hasWakeWord) return;

        this.isActive = true;
        this.activeUser = userId;
        console.log(`[Voice] Session activated by ${userId}`);

        // Remove wake word from text for processing
        let cleanText = lower;
        for (const w of WAKE_WORDS) {
          cleanText = cleanText.replace(w, "").trim();
        }

        // If there's remaining text after wake word, process it
        if (cleanText.length > 2) {
          await this.respond(cleanText);
        } else {
          await this.respond("greeting_only");
        }
        return;
      }

      // Active conversation — check for goodbye
      const lower = text.toLowerCase();
      if (lower.includes("bye meera") || lower.includes("thanks meera") || lower.includes("that's all")) {
        await this.respond("goodbye");
        this.deactivate();
        return;
      }

      // Reset silence timer and respond
      this.resetSilenceTimer();
      await this.respond(text);
    } catch (err) {
      console.error("[Voice] Process error:", err.message);
      this.cleanupFile(pcmPath);
    }
  }

  async speechToText(wavPath) {
    /**
     * Uses Groq's Whisper API for STT.
     * Fast, free tier available, great accuracy.
     */
    const Groq = require("groq-sdk");
    const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
    const { createReadStream } = require("fs");

    try {
      const transcription = await groq.audio.transcriptions.create({
        file: createReadStream(wavPath),
        model: "whisper-large-v3",
        language: "en",
        response_format: "text",
      });
      return transcription;
    } catch (err) {
      console.error("[STT] Error:", err.message);
      return null;
    }
  }

  async respond(userText) {
    this.isListening = false;
    this.isSpeaking = true;

    try {
      let aiResponse;

      if (userText === "greeting_only") {
        aiResponse = "Hey! I'm here. What's up?";
      } else if (userText === "goodbye") {
        aiResponse = "See you later! Keep coding!";
      } else {
        // Get AI response
        aiResponse = await chat(userText);
        // Keep voice responses short
        if (aiResponse.length > 500) {
          aiResponse = aiResponse.substring(0, 500).replace(/\n/g, " ").trim();
        }
      }

      console.log(`[Voice] Meera: "${aiResponse.substring(0, 100)}..."`);

      // TTS — generate audio
      const audioPath = await this.textToSpeech(aiResponse);
      if (!audioPath) {
        this.isSpeaking = false;
        return;
      }

      // Play audio in VC
      const resource = createAudioResource(audioPath);
      this.player.play(resource);

      // Cleanup after playing
      this.player.once(AudioPlayerStatus.Idle, () => {
        this.cleanupFile(audioPath);
      });
    } catch (err) {
      console.error("[Voice] Respond error:", err.message);
      this.isSpeaking = false;
    }
  }

  async textToSpeech(text) {
    /**
     * TTS Pipeline:
     * 1. First: Use local VITS model if available
     * 2. Fallback: Use system TTS (say command on macOS) for development
     *
     * In production, this will use the trained MeeraVITS model.
     * For now, using a placeholder that generates audio via system TTS.
     */
    const outputPath = join(TEMP_DIR, `tts_${Date.now()}.wav`);

    try {
      // Try local VITS model first
      const vitsPath = join(__dirname, "..", "..", "training", "voice", "checkpoints", "best.pt");
      if (existsSync(vitsPath)) {
        // Call Python inference script
        const scriptPath = join(__dirname, "..", "..", "training", "voice", "scripts", "voice_inference.py");
        execSync(
          `python3 -c "
import sys; sys.path.insert(0, '${join(__dirname, "..", "..", "training", "voice", "scripts")}')
from voice_inference import load_model, synthesize, save_audio
model = load_model()
audio = synthesize(model, '''${text.replace(/'/g, "\\'")}''')
import soundfile as sf
sf.write('${outputPath}', audio, 22050)
" 2>/dev/null`
        );
        if (existsSync(outputPath)) return outputPath;
      }

      // Fallback: macOS say command (for development)
      const aiffPath = outputPath.replace(".wav", ".aiff");
      execSync(`say -v "Samantha" -o "${aiffPath}" "${text.replace(/"/g, '\\"').substring(0, 300)}"`);
      execSync(`ffmpeg -y -i "${aiffPath}" -ar 48000 -ac 2 "${outputPath}" 2>/dev/null`);
      this.cleanupFile(aiffPath);

      return existsSync(outputPath) ? outputPath : null;
    } catch (err) {
      console.error("[TTS] Error:", err.message);
      return null;
    }
  }

  resetSilenceTimer() {
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => {
      if (this.isActive) {
        console.log("[Voice] Silence timeout — deactivating session");
        this.deactivate();
      }
    }, SILENCE_TIMEOUT);
  }

  deactivate() {
    this.isActive = false;
    this.activeUser = null;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    // Keep connection alive but stop active listening
    console.log("[Voice] Session deactivated (still in VC, waiting for wake word)");
  }

  destroy() {
    this.isActive = false;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.listeners.forEach(({ audioStream }) => audioStream.destroy());
    this.listeners.clear();
    this.connection.destroy();
    sessions.delete(this.guild.id);
    console.log(`[Voice] Left VC in ${this.guild.name}`);
  }

  cleanupFile(path) {
    try {
      if (existsSync(path)) unlinkSync(path);
    } catch {}
  }
}

/**
 * Join a voice channel and start listening for wake word.
 */
function joinVC(voiceChannel) {
  if (sessions.has(voiceChannel.guild.id)) {
    return sessions.get(voiceChannel.guild.id);
  }

  const connection = joinVoiceChannel({
    channelId: voiceChannel.id,
    guildId: voiceChannel.guild.id,
    adapterCreator: voiceChannel.guild.voiceAdapterCreator,
    selfDeaf: false, // need to hear users
    selfMute: false,
  });

  const session = new VoiceSession(connection, voiceChannel, voiceChannel.guild);
  sessions.set(voiceChannel.guild.id, session);

  connection.on(VoiceConnectionStatus.Ready, () => {
    console.log(`[Voice] Connected to ${voiceChannel.name} in ${voiceChannel.guild.name}`);
    session.startListening();
  });

  connection.on(VoiceConnectionStatus.Disconnected, () => {
    session.destroy();
  });

  return session;
}

/**
 * Leave voice channel.
 */
function leaveVC(guildId) {
  const session = sessions.get(guildId);
  if (session) session.destroy();
}

/**
 * Check if bot is in a VC in this guild.
 */
function isInVC(guildId) {
  return sessions.has(guildId);
}

module.exports = { joinVC, leaveVC, isInVC, sessions };
