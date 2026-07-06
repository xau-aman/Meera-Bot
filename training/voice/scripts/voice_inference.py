"""Meera Voice Inference — Generate speech from text using trained VITS model."""
import os
import torch
import soundfile as sf
import numpy as np
from pathlib import Path

from vits_model import MeeraVITS

BASE_DIR = Path(__file__).parent.parent
CKPT_PATH = BASE_DIR / "checkpoints" / "best.pt"
OUTPUT_DIR = BASE_DIR / "output"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 22050


def text_to_sequence(text):
    """Convert text to integer sequence."""
    chars = " abcdefghijklmnopqrstuvwxyz'.,!?-"
    text = text.lower().strip()
    return [chars.index(c) + 1 if c in chars else 0 for c in text]


def load_model():
    """Load trained MeeraVITS model."""
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]
    mc = config["model"]

    model = MeeraVITS(
        n_vocab=256,
        spec_channels=config["data"]["n_mel_channels"],
        inter_channels=mc["inter_channels"],
        hidden_channels=mc["hidden_channels"],
        filter_channels=mc["filter_channels"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        kernel_size=mc["kernel_size"],
        p_dropout=mc["p_dropout"],
        resblock_kernel_sizes=mc["resblock_kernel_sizes"],
        resblock_dilation_sizes=mc["resblock_dilation_sizes"],
        upsample_rates=mc["upsample_rates"],
        upsample_initial_channel=mc["upsample_initial_channel"],
        upsample_kernel_sizes=mc["upsample_kernel_sizes"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.dec.remove_weight_norm()

    print(f"Loaded MeeraVITS ({model.count_params() / 1e6:.1f}M params) from step {checkpoint['step']}")
    return model


@torch.no_grad()
def synthesize(model, text, noise_scale=0.667, length_scale=1.0):
    """Generate audio from text."""
    seq = text_to_sequence(text)
    x = torch.LongTensor([seq]).to(DEVICE)
    x_lengths = torch.LongTensor([len(seq)]).to(DEVICE)

    audio = model.infer(x, x_lengths, noise_scale=noise_scale, length_scale=length_scale)
    audio = audio.squeeze().cpu().numpy()

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.95
    return audio


def save_audio(audio, filename):
    """Save audio to file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = OUTPUT_DIR / filename
    sf.write(path, audio, SAMPLE_RATE)
    print(f"Saved: {path}")
    return path


def main():
    model = load_model()
    print(f"\nMeera Voice ready on {DEVICE}! Type 'quit' to exit.\n")

    while True:
        text = input("Text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        audio = synthesize(model, text)
        filename = f"meera_{len(os.listdir(OUTPUT_DIR)) if OUTPUT_DIR.exists() else 0:04d}.wav"
        save_audio(audio, filename)
        print(f"Duration: {len(audio) / SAMPLE_RATE:.2f}s\n")


if __name__ == "__main__":
    main()
