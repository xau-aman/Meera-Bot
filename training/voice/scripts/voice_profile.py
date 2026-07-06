"""Fine-tune Moshi's voice for Meera's personality.

Moshi supports voice conditioning — we can guide its output voice
by providing reference audio clips of the desired voice style.

This script:
1. Prepares voice reference clips (Indian English female)
2. Creates voice embeddings for Moshi conditioning
3. Saves the voice profile for inference

You need 5-10 minutes of clean reference audio.
"""
import os
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REFERENCE_DIR = BASE_DIR / "data" / "voice_reference"
OUTPUT_DIR = BASE_DIR / "model" / "voice_profile"
CONFIG_PATH = BASE_DIR / "configs" / "moshi_config.json"


def prepare_reference_audio():
    """Instructions for preparing voice reference clips."""
    os.makedirs(REFERENCE_DIR, exist_ok=True)

    print("=" * 50)
    print("  MEERA VOICE PROFILE SETUP")
    print("=" * 50)
    print(f"""
To give Meera a custom voice, you need reference audio clips.

OPTIONS:

1. Record your own (best quality):
   - Find a female speaker with Indian English accent
   - Record 5-10 minutes of clear speech
   - Split into 10-30 second clips
   - Save as WAV (24kHz, mono)
   - Place in: {REFERENCE_DIR}/

2. Use open-source Indian English voice:
   - IITM TTS Database: https://www.iitm.ac.in/donlab/tts/database.php
   - Download English Female speaker
   - Extract 10-20 clean clips
   - Place in: {REFERENCE_DIR}/

3. Use AI-generated reference (quick start):
   - Use ElevenLabs/PlayHT to generate a few clips
   - with the voice style you want for Meera
   - Save as WAV in: {REFERENCE_DIR}/

FILE FORMAT:
   - WAV, 24000Hz, mono, 16-bit PCM
   - 10-30 seconds each
   - Clean audio (no background noise)
   - Natural conversational speech

After placing files, run:
   python voice_profile.py --create
""")


def create_voice_profile():
    """Create voice embedding from reference clips."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = list(REFERENCE_DIR.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files found in {REFERENCE_DIR}")
        print("Run: python voice_profile.py  (for instructions)")
        return

    print(f"Found {len(wav_files)} reference clips")

    try:
        import mlx.core as mx
        from moshi_mlx import models
        import soundfile as sf

        # Load model for voice encoding
        model_path = BASE_DIR / "model" / "moshika-mlx-bf16"
        if not model_path.exists():
            print("ERROR: Moshi model not found. Run: python scripts/setup_moshi.py")
            return

        print("Loading Moshi model for voice encoding...")
        model = models.load_model(model_path)

        # Extract voice embeddings from reference clips
        embeddings = []
        for wav_file in wav_files:
            audio, sr = sf.read(wav_file)
            if sr != 24000:
                # Resample to 24kHz
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)

            # Get voice embedding
            audio_tensor = mx.array(audio.astype(np.float32).reshape(1, -1))
            embedding = model.encode_voice(audio_tensor)
            embeddings.append(np.array(embedding))
            print(f"  Processed: {wav_file.name}")

        # Average embeddings for stable voice profile
        voice_profile = np.mean(embeddings, axis=0)

        # Save
        profile_path = OUTPUT_DIR / "meera_voice.npy"
        np.save(profile_path, voice_profile)

        # Save metadata
        meta = {
            "name": "Meera",
            "description": "Warm, friendly Indian English female voice",
            "num_reference_clips": len(wav_files),
            "embedding_shape": list(voice_profile.shape),
        }
        with open(OUTPUT_DIR / "profile_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n✅ Voice profile saved to: {profile_path}")
        print(f"   Embedding shape: {voice_profile.shape}")
        print(f"\n   Meera now has her own voice! Start the server:")
        print(f"   python scripts/moshi_server.py")

    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install moshi_mlx soundfile librosa")

    except AttributeError:
        # If model doesn't have encode_voice, use alternative approach
        print("\nNote: Direct voice encoding not available in this Moshi version.")
        print("Using voice conditioning via prompt audio instead.")

        # Alternative: save reference audio paths for runtime conditioning
        profile = {
            "name": "Meera",
            "reference_clips": [str(f) for f in wav_files[:5]],
            "conditioning_method": "prompt_audio",
        }
        with open(OUTPUT_DIR / "meera_voice.json", "w") as f:
            json.dump(profile, f, indent=2)

        print(f"✅ Voice profile (prompt-based) saved!")


if __name__ == "__main__":
    import sys

    if "--create" in sys.argv:
        create_voice_profile()
    else:
        prepare_reference_audio()
