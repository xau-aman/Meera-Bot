"""Setup Moshi speech-to-speech model for Meera.

Moshi by Kyutai — real-time full-duplex speech-to-speech model.
Runs on Apple Silicon (MLX) with ~200ms latency.

Usage:
    python setup_moshi.py          # Download model
    python setup_moshi.py --test   # Test inference
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "model"


def download_model():
    """Download Moshi model weights from HuggingFace."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 50)
    print("  MOSHI MODEL SETUP — Meera Voice")
    print("=" * 50)

    try:
        from huggingface_hub import snapshot_download

        print("\nDownloading Moshi MLX model (Apple Silicon optimized)...")
        print("This is ~3.5GB, might take a few minutes.\n")

        snapshot_download(
            repo_id="kyutai/moshika-mlx-q4",
            local_dir=str(MODEL_DIR / "moshika-mlx-q4"),
        )

        print(f"\n✅ Model downloaded to: {MODEL_DIR / 'moshika-mlx-q4'}")
        print("\nNext steps:")
        print("  1. python moshi_server.py     # Start Moshi server")
        print("  2. npm start                  # Start Discord bot")
        print("  3. /voice join                # Join VC and say 'Hey Meera'")

    except ImportError:
        print("ERROR: huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
        sys.exit(1)


def test_model():
    """Quick test to verify model loads correctly."""
    print("Testing Moshi model load...")

    try:
        import mlx.core as mx
        from moshi_mlx import models

        model_path = MODEL_DIR / "moshika-mlx-q4"
        if not model_path.exists():
            print("ERROR: Model not found. Run: python setup_moshi.py")
            sys.exit(1)

        print(f"Loading from {model_path}...")
        model = models.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"   Ready for real-time speech-to-speech inference.")

    except Exception as e:
        print(f"ERROR: {e}")
        print("\nMake sure you have:")
        print("  pip install moshi_mlx")
        sys.exit(1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_model()
    else:
        download_model()
