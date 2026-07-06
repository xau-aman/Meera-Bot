"""Meera Voice Server — Real-time speech-to-speech using Moshi (MLX).

Uses the correct moshi_mlx API:
- rustymimi for audio tokenization (encode/decode PCM ↔ tokens)
- LmGen for language model inference (audio tokens → response tokens)

Architecture:
    Discord Bot (Node.js) ←WebSocket→ This Server (Python)
    
    Audio In → rustymimi.encode → LmGen.step → rustymimi.decode → Audio Out

Usage:
    python moshi_server.py [--port 8765]
"""
import asyncio
import json
import sys
import os
import time
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import websockets
import rustymimi
import huggingface_hub

from moshi_mlx import models, utils

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model" / "moshika-mlx-q4"

HOST = "127.0.0.1"
PORT = int(os.environ.get("MOSHI_PORT", 8765))
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms at 24kHz


def get_model_files():
    """Get paths to model files (download from HF if needed)."""
    model_file = str(MODEL_DIR / "model.q4.safetensors")
    tokenizer_file = str(MODEL_DIR / "tokenizer_spm_32k_3.model")
    mimi_file = str(MODEL_DIR / "tokenizer-e351c8d8-checkpoint125.safetensors")

    # Check if files exist
    for f in [model_file, tokenizer_file, mimi_file]:
        if not os.path.exists(f):
            print(f"ERROR: Missing file: {f}")
            print("Run: python scripts/setup_moshi.py")
            sys.exit(1)

    return model_file, tokenizer_file, mimi_file


class MoshiEngine:
    """Moshi inference engine — handles audio-to-audio generation."""

    def __init__(self):
        self.gen = None
        self.audio_tokenizer = None
        self.text_tokenizer = None
        self.is_ready = False

    def load(self):
        """Load all model components."""
        model_file, tokenizer_file, mimi_file = get_model_files()

        print("[Engine] Loading text tokenizer...")
        import sentencepiece
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_file)

        print("[Engine] Loading Moshi LM (q4 quantized)...")
        mx.random.seed(299792458)
        lm_config = models.config_v0_1()
        model = models.Lm(lm_config)
        model.set_dtype(mx.bfloat16)
        nn.quantize(model, bits=4, group_size=32)
        model.load_weights(model_file, strict=True)
        model.warmup()

        print("[Engine] Creating generator...")
        self.gen = models.LmGen(
            model=model,
            max_steps=2048,
            text_sampler=utils.Sampler(),
            audio_sampler=utils.Sampler(),
            check=False,
        )

        print("[Engine] Loading audio tokenizer (Mimi)...")
        self.audio_tokenizer = rustymimi.StreamTokenizer(mimi_file)

        # Warmup audio tokenizer
        print("[Engine] Warming up...")
        for _ in range(4):
            silence = np.zeros(FRAME_SIZE, dtype=np.float32)
            self.audio_tokenizer.encode(silence)
            time.sleep(0.01)
            data = None
            while data is None:
                time.sleep(0.005)
                data = self.audio_tokenizer.get_encoded()

        self.is_ready = True
        print("[Engine] ✅ Ready!")

    def process_frame(self, pcm_frame):
        """Process one audio frame and return response audio.

        Args:
            pcm_frame: numpy float32 array, 1920 samples at 24kHz (80ms)

        Returns:
            Response PCM audio (float32) or None
        """
        if not self.is_ready or self.gen is None:
            return None

        # Encode input audio to tokens
        self.audio_tokenizer.encode(pcm_frame)
        encoded = self.audio_tokenizer.get_encoded()

        if encoded is None:
            return None

        # Run LM step
        tokens = mx.array(encoded).transpose(1, 0)[:, :8]
        text_token = self.gen.step(tokens)

        # Get response audio tokens
        audio_tokens = self.gen.last_audio_tokens()
        if audio_tokens is None:
            return None

        # Decode audio tokens back to PCM
        audio_tokens_np = np.array(audio_tokens).astype(np.uint32)
        self.audio_tokenizer.decode(audio_tokens_np)

        # Get decoded audio
        decoded = self.audio_tokenizer.get_decoded()
        return decoded

    def reset(self):
        """Reset for new conversation."""
        if self.gen is not None:
            # Recreate generator for fresh state
            pass


# Global engine
engine = MoshiEngine()


async def handle_connection(websocket):
    """Handle WebSocket connection from Discord bot."""
    addr = websocket.remote_address
    print(f"[Server] Client connected: {addr}")

    try:
        async for message in websocket:
            # Control messages (JSON string)
            if isinstance(message, str):
                data = json.loads(message)
                cmd = data.get("cmd")

                if cmd == "start":
                    await websocket.send(json.dumps({"status": "ready"}))
                    print("[Server] Session started")

                elif cmd == "stop":
                    await websocket.send(json.dumps({"status": "stopped"}))
                    print("[Server] Session stopped")

                elif cmd == "ping":
                    await websocket.send(json.dumps({"status": "pong"}))

            # Audio data (binary) — PCM float32 at 24kHz
            elif isinstance(message, bytes):
                # Convert bytes to float32 numpy array
                pcm_input = np.frombuffer(message, dtype=np.float32)

                # Process in FRAME_SIZE chunks
                for i in range(0, len(pcm_input), FRAME_SIZE):
                    chunk = pcm_input[i:i + FRAME_SIZE]
                    if len(chunk) < FRAME_SIZE:
                        # Pad short chunk
                        chunk = np.pad(chunk, (0, FRAME_SIZE - len(chunk)))

                    response = engine.process_frame(chunk)

                    if response is not None:
                        # Send response audio back
                        await websocket.send(response.astype(np.float32).tobytes())

    except websockets.exceptions.ConnectionClosed:
        print(f"[Server] Client disconnected: {addr}")
    except Exception as e:
        print(f"[Server] Error: {e}")


async def main():
    # Load model
    engine.load()

    print(f"\n🎙️  Meera Voice Server running on ws://{HOST}:{PORT}")
    print(f"   Model: Moshika (female voice, q4 quantized)")
    print(f"   Latency: ~200ms")
    print(f"   Waiting for Discord bot connection...\n")

    async with websockets.serve(handle_connection, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        PORT = int(sys.argv[idx + 1])

    asyncio.run(main())
