# Meera Voice — Real-time Speech-to-Speech

Meera uses **Moshi** (by Kyutai) for real-time voice conversations in Discord.  
No STT/TTS pipeline. Direct audio-in → audio-out. **~200ms latency.**

## Architecture

```
User speaks in Discord VC
    │
    ▼
Discord Bot (Node.js) ──WebSocket──► Moshi Server (Python/MLX)
    │                                      │
    │  audio stream (24kHz PCM)            │ Moshi processes speech
    │                                      │ generates response speech
    ◄──────────────────────────────────────┘
    │
    ▼
Bot speaks response in VC (real-time streaming)
```

## Setup

### 1. Install Python dependencies

```bash
cd training/voice
pip install -r requirements.txt
```

### 2. Download Moshi model (~3.5GB)

```bash
python scripts/setup_moshi.py
```

### 3. (Optional) Create custom voice profile

```bash
python scripts/voice_profile.py          # See instructions
python scripts/voice_profile.py --create # After adding reference audio
```

### 4. Start Moshi server

```bash
python scripts/moshi_server.py
```

### 5. Start Discord bot (in another terminal)

```bash
cd ../../
npm install
npm start
```

### 6. Use in Discord

- `/voice join` or click 🎙️ Join VC button
- Say **"Hey Meera"** to start conversation
- Talk naturally — Meera responds in real-time
- Say **"Bye Meera"** or stay silent 30s to end

## Custom Voice

To give Meera a unique voice:

1. Get 5-10 minutes of reference audio (Indian English female)
2. Place WAV files in `data/voice_reference/`
3. Run `python scripts/voice_profile.py --create`

## Requirements

- **macOS with M4** (Apple Silicon) — uses MLX for fast inference
- **~4GB RAM** for model
- **ffmpeg** installed (`brew install ffmpeg`)
- **Python 3.10+**

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Moshi server not running" | Start: `python scripts/moshi_server.py` |
| High latency | Ensure MLX is using GPU: `pip install mlx --upgrade` |
| No audio output | Check bot isn't self-deafened in Discord |
| Wake word not detecting | Speak clearly, ensure GROQ_API_KEY is set |
