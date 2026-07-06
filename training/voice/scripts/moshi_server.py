"""Meera Voice Server — Real-time speech-to-speech using Moshi.

This runs as a local WebSocket server that the Discord bot connects to.
Audio streams in, Meera's voice streams out. ~200ms latency.

Architecture:
    Discord Bot (Node.js) <--WebSocket--> Moshi Server (Python/MLX)

Usage:
    python moshi_server.py [--port 8765]
"""
import asyncio
import json
import struct
import sys
import os
from pathlib import Path

import numpy as np
import websockets

# Add parent to path for imports
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model" / "moshika-mlx-bf16"

# Server config
HOST = "127.0.0.1"
PORT = int(os.environ.get("MOSHI_PORT", 8765))
SAMPLE_RATE = 24000  # Moshi uses 24kHz
FRAME_SIZE = 1920  # 80ms frames at 24kHz


class MoshiSession:
    """Manages a single Moshi inference session."""

    def __init__(self):
        self.model = None
        self.is_active = False
        self.audio_buffer = bytearray()

    async def initialize(self):
        """Load Moshi model."""
        try:
            import mlx.core as mx
            from moshi_mlx import models

            if not MODEL_DIR.exists():
                print("ERROR: Model not found. Run: python scripts/setup_moshi.py")
                sys.exit(1)

            print(f"Loading Moshi model from {MODEL_DIR}...")
            self.model = models.load_model(MODEL_DIR)
            self.is_active = True
            print("✅ Moshi model loaded — ready for conversations!")
            return True

        except ImportError as e:
            print(f"ERROR: Missing dependency — {e}")
            print("Run: pip install -r requirements.txt")
            return False

    async def process_audio(self, audio_bytes):
        """Process incoming audio and generate response audio.

        Args:
            audio_bytes: Raw PCM audio (24kHz, mono, int16)

        Returns:
            Response audio bytes (24kHz, mono, int16) or None
        """
        if not self.model or not self.is_active:
            return None

        try:
            import mlx.core as mx

            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Feed audio to Moshi and get response
            # Moshi processes in streaming fashion — feed chunks, get chunks back
            input_tensor = mx.array(audio_np.reshape(1, -1))

            # Run one step of inference
            output = self.model.step(input_tensor)

            if output is not None:
                # Convert back to int16 PCM
                output_np = np.array(output).flatten()
                output_np = np.clip(output_np * 32768.0, -32768, 32767).astype(np.int16)
                return output_np.tobytes()

            return None

        except Exception as e:
            print(f"[Moshi] Inference error: {e}")
            return None

    def reset(self):
        """Reset session state for new conversation."""
        if self.model:
            try:
                self.model.reset()
            except:
                pass
        self.audio_buffer = bytearray()


# Global session
session = MoshiSession()


async def handle_connection(websocket):
    """Handle a WebSocket connection from the Discord bot."""
    client_addr = websocket.remote_address
    print(f"[Server] Client connected: {client_addr}")

    session.reset()

    try:
        async for message in websocket:
            # Handle control messages (JSON)
            if isinstance(message, str):
                data = json.loads(message)
                cmd = data.get("cmd")

                if cmd == "start":
                    session.reset()
                    await websocket.send(json.dumps({"status": "ready"}))
                    print(f"[Server] Session started")

                elif cmd == "stop":
                    session.reset()
                    await websocket.send(json.dumps({"status": "stopped"}))
                    print(f"[Server] Session stopped")

                elif cmd == "ping":
                    await websocket.send(json.dumps({"status": "pong"}))

            # Handle audio data (binary)
            elif isinstance(message, bytes):
                # Process audio through Moshi
                response_audio = await session.process_audio(message)

                if response_audio:
                    # Send response audio back to bot
                    await websocket.send(response_audio)

    except websockets.exceptions.ConnectionClosed:
        print(f"[Server] Client disconnected: {client_addr}")
    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        session.reset()


async def main():
    # Initialize model
    success = await session.initialize()
    if not success:
        sys.exit(1)

    # Start WebSocket server
    print(f"\n🎙️  Meera Voice Server running on ws://{HOST}:{PORT}")
    print(f"   Waiting for Discord bot connection...\n")

    async with websockets.serve(handle_connection, HOST, PORT):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        PORT = int(sys.argv[idx + 1])

    asyncio.run(main())
