/**
 * Meera Voice Handler — Real-time Speech-to-Speech via Moshi
 *
 * Flow:
 *   User speaks in VC → audio stream → Moshi server (WebSocket) → response audio → bot speaks
 *
 * No STT/TTS pipeline. Direct audio-in, audio-out. ~200ms latency.
 *
 * Wake word: "Hey Meera" activates session
 * Session stays active until "bye meera" or 30s silence
 */

const {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  VoiceConnectionStatus,
  EndBehaviorType,
  StreamType,
} = require("@discordjs/voice");
const { Readable, Transform, PassThrough } = require("stream");
const { createWriteStream, unlinkSync, existsSync, mkdirSync, createReadStream } = require("fs");
const { join } = require("path");
const { execSync } = require("child_process");
const WebSocket = require("ws");
const prism = require("prism-media");

// Config
const MOSHI_URL = process.env.MOSHI_URL || "ws://127.0.0.1:8765";
const SILENCE_TIMEOUT = 30000;
const WAKE_WORDS = ["hey meera", "hi meera", "meera"];
const TEMP_DIR = join(__dirname, "..", "..", "temp");

if (!existsSync(TEMP_DIR)) mkdirSync(TEMP_DIR, { recursive: true });

// Active sessions per guild
const sessions = new Map();

class VoiceSession {
  constructor(connection, channel, guild) {
    this.connection = connection;
    this.channel = channel;
    this.guild = guild;
    this.player = createAudioPlayer();
    this.ws = null;
    this.isActive = false; // true after wake word
    this.isSpeaking = false;
    this.silenceTimer = null;
    this.responseBuffer = Buffer.alloc(0);
    this.audioPassthrough = null;

    connection.subscribe(this.player);

    this.player.on(AudioPlayerStatus.Idle, () => {
      this.isSpeaking = false;
    });

    this.player.on("error", (err) => {
      console.error("[Voice] Player error:", err.message);
      this.isSpeaking = false;
    });
  }

  async connect() {
    /**
     * Connect to Moshi WebSocket server.
     * The server handles all AI inference locally.
     */
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(MOSHI_URL);

      this.ws.on("open", () => {
        console.log("[Voice] Connected to Moshi server");
        this.ws.send(JSON.stringify({ cmd: "start" }));
        resolve();
      });

      this.ws.on("message", (data) => {
        if (typeof data === "string") {
          // Control message
          const msg = JSON.parse(data);
          if (msg.status === "ready") {
            console.log("[Voice] Moshi session ready");
          }
          return;
        }

        // Binary audio response from Moshi — play it immediately
        if (this.isActive && !this.isSpeaking) {
          this.playAudioChunk(data);
        }
      });

      this.ws.on("error", (err) => {
        console.error("[Voice] Moshi WebSocket error:", err.message);
        reject(err);
      });

      this.ws.on("close", () => {
        console.log("[Voice] Moshi connection closed");
        this.ws = null;
      });
    });
  }

  startListening() {
    const receiver = this.connection.receiver;

    receiver.speaking.on("start", (userId) => {
      if (this.isSpeaking) return;
      this.listenToUser(userId);
    });
  }

  listenToUser(userId) {
    const receiver = this.connection.receiver;

    const audioStream = receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 1000 },
    });

    // Decode Opus → PCM (48kHz, mono, 16-bit)
    const decoder = new prism.opus.Decoder({ rate: 48000, channels: 1, frameSize: 960 });
    const chunks = [];

    const pcmStream = audioStream.pipe(decoder);

    pcmStream.on("data", (chunk) => {
      chunks.push(chunk);
    });

    pcmStream.on("end", async () => {
      const fullAudio = Buffer.concat(chunks);
      if (fullAudio.length < 4800) return; // too short, ignore

      if (!this.isActive) {
        // Check for wake word using Groq Whisper (only for detection)
        const text = await this.quickSTT(fullAudio);
        if (!text) return;

        const lower = text.toLowerCase();
        const hasWakeWord = WAKE_WORDS.some((w) => lower.includes(w));
        if (!hasWakeWord) return;

        console.log(`[Voice] Wake word detected! Activating session.`);
        this.isActive = true;
        this.resetSilenceTimer();

        // Send the audio to Moshi to start conversation
        await this.sendToMoshi(fullAudio);
      } else {
        // Active session — stream directly to Moshi
        this.resetSilenceTimer();
        await this.sendToMoshi(fullAudio);
      }
    });
  }

  async sendToMoshi(pcmBuffer) {
    /**
     * Send audio to Moshi server.
     * Resample from 48kHz int16 to 24kHz float32 (Moshi's native format).
     */
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      try {
        await this.connect();
      } catch {
        return;
      }
    }

    // Convert int16 to float32 and resample 48kHz → 24kHz
    const samples16 = new Int16Array(pcmBuffer.buffer, pcmBuffer.byteOffset, pcmBuffer.length / 2);
    const resampled = new Float32Array(Math.floor(samples16.length / 2));
    for (let i = 0; i < resampled.length; i++) {
      resampled[i] = samples16[i * 2] / 32768.0;
    }

    // Send float32 PCM to Moshi
    this.ws.send(Buffer.from(resampled.buffer));
  }

  playAudioChunk(audioData) {
    /**
     * Play Moshi's response audio in the VC.
     * Audio comes as 24kHz float32 PCM — convert to 48kHz int16 for Discord.
     */
    this.isSpeaking = true;

    // Convert float32 to int16 and upsample 24kHz → 48kHz
    const samples = new Float32Array(audioData.buffer, audioData.byteOffset, audioData.length / 4);
    const upsampled = new Int16Array(samples.length * 2);
    for (let i = 0; i < samples.length; i++) {
      const val = Math.max(-1, Math.min(1, samples[i]));
      const sample16 = Math.round(val * 32767);
      upsampled[i * 2] = sample16;
      upsampled[i * 2 + 1] = i < samples.length - 1
        ? Math.round((samples[i] + samples[i + 1]) / 2 * 32767)
        : sample16;
    }

    // Create readable stream from buffer
    const audioBuffer = Buffer.from(upsampled.buffer);
    const readable = new Readable({
      read() {
        this.push(audioBuffer);
        this.push(null);
      },
    });

    const resource = createAudioResource(readable, {
      inputType: StreamType.Raw,
      inlineVolume: false,
    });

    this.player.play(resource);
  }

  async quickSTT(pcmBuffer) {
    /**
     * Quick STT for wake word detection only.
     * Uses Groq Whisper — fast and free.
     * Only used for "Hey Meera" detection, not for conversation.
     */
    const wavPath = join(TEMP_DIR, `wake_${Date.now()}.wav`);

    try {
      // Write PCM to WAV file
      this.pcmToWav(pcmBuffer, wavPath, 48000);

      const Groq = require("groq-sdk");
      const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

      const transcription = await groq.audio.transcriptions.create({
        file: createReadStream(wavPath),
        model: "whisper-large-v3",
        language: "en",
        response_format: "text",
      });

      this.cleanupFile(wavPath);
      return transcription;
    } catch (err) {
      this.cleanupFile(wavPath);
      return null;
    }
  }

  pcmToWav(pcmBuffer, outputPath, sampleRate) {
    /**
     * Convert raw PCM buffer to WAV file.
     */
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcmBuffer.length;
    const headerSize = 44;

    const header = Buffer.alloc(headerSize);
    header.write("RIFF", 0);
    header.writeUInt32LE(dataSize + headerSize - 8, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16); // fmt chunk size
    header.writeUInt16LE(1, 20); // PCM format
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(byteRate, 28);
    header.writeUInt16LE(blockAlign, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write("data", 36);
    header.writeUInt32LE(dataSize, 40);

    const { writeFileSync } = require("fs");
    writeFileSync(outputPath, Buffer.concat([header, pcmBuffer]));
  }

  resetSilenceTimer() {
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => {
      if (this.isActive) {
        console.log("[Voice] Silence timeout — session deactivated");
        this.deactivate();
      }
    }, SILENCE_TIMEOUT);
  }

  deactivate() {
    this.isActive = false;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this.ws) {
      this.ws.send(JSON.stringify({ cmd: "stop" }));
    }
  }

  destroy() {
    this.deactivate();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.connection.destroy();
    sessions.delete(this.guild.id);
    console.log(`[Voice] Left VC in ${this.guild.name}`);
  }

  cleanupFile(path) {
    try { if (existsSync(path)) unlinkSync(path); } catch {}
  }
}

/**
 * Join a voice channel and start listening.
 */
async function joinVC(voiceChannel) {
  if (sessions.has(voiceChannel.guild.id)) {
    return sessions.get(voiceChannel.guild.id);
  }

  const connection = joinVoiceChannel({
    channelId: voiceChannel.id,
    guildId: voiceChannel.guild.id,
    adapterCreator: voiceChannel.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
  });

  const session = new VoiceSession(connection, voiceChannel, voiceChannel.guild);
  sessions.set(voiceChannel.guild.id, session);

  connection.on(VoiceConnectionStatus.Ready, async () => {
    console.log(`[Voice] Connected to ${voiceChannel.name}`);

    // Connect to Moshi server
    try {
      await session.connect();
    } catch (err) {
      console.log("[Voice] Moshi server not running — wake word detection only");
    }

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
 * Check if bot is in VC.
 */
function isInVC(guildId) {
  return sessions.has(guildId);
}

module.exports = { joinVC, leaveVC, isInVC, sessions };
