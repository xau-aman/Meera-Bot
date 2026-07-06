"""Download and preprocess Indian English female voice dataset for VITS training."""
import os
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "voice" / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FILELIST_DIR = BASE_DIR / "voice" / "filelists"

SAMPLE_RATE = 22050

# IITM Indian English TTS dataset (female speaker)
# Alternative: IndicTTS dataset from IIT Madras
DATASET_URLS = {
    "iitm_english": "https://www.iitm.ac.in/donlab/tts/database/english/english_female.zip",
}


def download_dataset():
    """Download Indian English female voice dataset."""
    os.makedirs(RAW_DIR, exist_ok=True)

    print("=" * 50)
    print("DATASET DOWNLOAD OPTIONS")
    print("=" * 50)
    print("""
Option 1: IITM Indian English Female (Recommended)
    - Download from: https://www.iitm.ac.in/donlab/tts/database.php
    - Select: English > Female
    - Place the extracted folder in: training/voice/data/raw/

Option 2: LJSpeech (English, American accent - fallback)
    - wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    - Extract to: training/voice/data/raw/

Option 3: Custom recordings
    - Record 1-2 hours of clear female speech
    - 22050Hz, mono, WAV format
    - One sentence per file
    - Place in: training/voice/data/raw/wavs/
    - Create metadata.csv: filename|transcription

After downloading, run this script again with --preprocess flag.
""")


def preprocess_audio():
    """Preprocess audio files: resample, normalize, trim silence."""
    import librosa
    import soundfile as sf
    import numpy as np

    os.makedirs(PROCESSED_DIR / "wavs", exist_ok=True)
    os.makedirs(FILELIST_DIR, exist_ok=True)

    # Find metadata file
    metadata_path = None
    for name in ["metadata.csv", "transcript.txt", "text.txt"]:
        p = RAW_DIR / name
        if p.exists():
            metadata_path = p
            break

    if not metadata_path:
        # Try to find any txt/csv in subdirectories
        for f in RAW_DIR.rglob("*.csv"):
            metadata_path = f
            break
        for f in RAW_DIR.rglob("*.txt"):
            if "readme" not in f.name.lower():
                metadata_path = f
                break

    if not metadata_path:
        print("ERROR: No metadata file found!")
        print("Expected: training/voice/data/raw/metadata.csv")
        print("Format: filename|transcription")
        return

    print(f"Using metadata: {metadata_path}")

    # Parse metadata
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                filename = parts[0].strip()
                text = parts[-1].strip()
                entries.append((filename, text))

    print(f"Found {len(entries)} entries")

    # Process audio files
    processed = []
    for filename, text in tqdm(entries, desc="Processing audio"):
        # Find the wav file
        wav_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            candidates = [
                RAW_DIR / "wavs" / f"{filename}{ext}",
                RAW_DIR / "wavs" / filename,
                RAW_DIR / f"{filename}{ext}",
                RAW_DIR / filename,
            ]
            for c in candidates:
                if c.exists():
                    wav_path = c
                    break
            if wav_path:
                break

        if not wav_path:
            continue

        try:
            # Load and resample
            audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE)

            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=23)

            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.95

            # Skip if too short or too long
            duration = len(audio) / SAMPLE_RATE
            if duration < 0.5 or duration > 15.0:
                continue

            # Save processed
            out_name = f"meera_{len(processed):05d}.wav"
            sf.write(PROCESSED_DIR / "wavs" / out_name, audio, SAMPLE_RATE)
            processed.append((out_name, text))

        except Exception as e:
            print(f"  Skipping {filename}: {e}")
            continue

    print(f"\nProcessed {len(processed)} files")

    # Create filelists (90/10 split)
    split_idx = int(len(processed) * 0.9)
    train_entries = processed[:split_idx]
    val_entries = processed[split_idx:]

    with open(FILELIST_DIR / "train.txt", "w") as f:
        for name, text in train_entries:
            f.write(f"data/processed/wavs/{name}|{text}\n")

    with open(FILELIST_DIR / "val.txt", "w") as f:
        for name, text in val_entries:
            f.write(f"data/processed/wavs/{name}|{text}\n")

    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")
    print("Done! Now run: python scripts/train_voice.py")


if __name__ == "__main__":
    import sys

    if "--preprocess" in sys.argv:
        preprocess_audio()
    else:
        download_dataset()
        print("\nAfter downloading, run:")
        print("  python scripts/prepare_voice_data.py --preprocess")
