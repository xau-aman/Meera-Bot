"""Train MeeraVITS — custom voice for Meera."""
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import librosa
import numpy as np

from vits_model import MeeraVITS, sequence_mask

BASE_DIR = Path(__file__).parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "meera_voice.json"
CKPT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ─── Phoneme mapping (simple char-level for now) ───
def text_to_sequence(text):
    """Convert text to integer sequence."""
    # Simple character-level encoding
    chars = " abcdefghijklmnopqrstuvwxyz'.,!?-"
    text = text.lower().strip()
    return [chars.index(c) + 1 if c in chars else 0 for c in text]


# ─── Dataset ───
class VoiceDataset(Dataset):
    def __init__(self, filelist_path, base_dir, config):
        self.base_dir = Path(base_dir)
        self.sr = config["data"]["sampling_rate"]
        self.n_fft = config["data"]["filter_length"]
        self.hop_length = config["data"]["hop_length"]
        self.win_length = config["data"]["win_length"]
        self.n_mels = config["data"]["n_mel_channels"]

        self.entries = []
        with open(filelist_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    self.entries.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wav_path, text = self.entries[idx]
        wav_path = self.base_dir / wav_path

        # Load audio
        audio, _ = librosa.load(wav_path, sr=self.sr)
        audio = torch.FloatTensor(audio)

        # Compute mel spectrogram
        mel = self._get_mel(audio)

        # Text to sequence
        text_seq = torch.LongTensor(text_to_sequence(text))

        return text_seq, mel, audio

    def _get_mel(self, audio):
        """Compute mel spectrogram."""
        audio_np = audio.numpy()
        mel = librosa.feature.melspectrogram(
            y=audio_np, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.win_length,
            n_mels=self.n_mels, fmin=0, fmax=8000
        )
        mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
        return torch.FloatTensor(mel)


def collate_fn(batch):
    """Pad sequences in batch."""
    text_seqs, mels, audios = zip(*batch)

    text_lengths = torch.LongTensor([len(t) for t in text_seqs])
    mel_lengths = torch.LongTensor([m.size(1) for m in mels])

    max_text_len = text_lengths.max()
    max_mel_len = mel_lengths.max()

    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(len(batch), mels[0].size(0), max_mel_len)

    for i, (t, m) in enumerate(zip(text_seqs, mels)):
        text_padded[i, :len(t)] = t
        mel_padded[i, :, :m.size(1)] = m

    return text_padded, text_lengths, mel_padded, mel_lengths


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print(f"Device: {DEVICE}")
    print(f"Config loaded from {CONFIG_PATH}")

    # Dataset
    filelist_dir = BASE_DIR / "filelists"
    train_ds = VoiceDataset(filelist_dir / "train.txt", BASE_DIR, config)
    val_ds = VoiceDataset(filelist_dir / "val.txt", BASE_DIR, config)

    train_loader = DataLoader(train_ds, batch_size=config["train"]["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
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

    print(f"MeeraVITS: {model.count_params() / 1e6:.1f}M params")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        betas=config["train"]["betas"],
        eps=config["train"]["eps"],
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["train"]["lr_decay"])
    writer = SummaryWriter(LOG_DIR)

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(config["train"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            text, text_len, mel, mel_len = [x.to(DEVICE) for x in batch]

            # Forward
            o, m_p, logs_p, m_q, logs_q, x_mask, y_mask, z = model(text, text_len, mel, mel_len)

            # Losses
            # Reconstruction loss (mel)
            mel_loss = F.l1_loss(o.squeeze(1)[:, :mel.size(2)], mel[:, 0, :]) * config["train"]["c_mel"]

            # KL divergence
            kl_loss = kl_divergence(m_p, logs_p, m_q, logs_q, x_mask) * config["train"]["c_kl"]

            loss = mel_loss + kl_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", mel=f"{mel_loss.item():.4f}", kl=f"{kl_loss.item():.4f}")

            # Log
            if global_step % config["train"]["log_interval"] == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/mel_loss", mel_loss.item(), global_step)
                writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)

            # Eval
            if global_step % config["train"]["eval_interval"] == 0:
                val_loss = evaluate(model, val_loader, config)
                writer.add_scalar("val/loss", val_loss, global_step)
                print(f"\n  Step {global_step} | Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        "model": model.state_dict(),
                        "config": config,
                        "step": global_step,
                    }, CKPT_DIR / "best.pt")
                    print(f"  ✅ Best model saved!")

        scheduler.step()

        # Save checkpoint every epoch
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": global_step,
            "epoch": epoch,
        }, CKPT_DIR / f"epoch_{epoch + 1}.pt")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    writer.close()


def evaluate(model, val_loader, config):
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            text, text_len, mel, mel_len = [x.to(DEVICE) for x in batch]
            o, m_p, logs_p, m_q, logs_q, x_mask, y_mask, z = model(text, text_len, mel, mel_len)

            mel_loss = F.l1_loss(o.squeeze(1)[:, :mel.size(2)], mel[:, 0, :])
            total_loss += mel_loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)


def kl_divergence(m_p, logs_p, m_q, logs_q, mask):
    """KL(q || p)"""
    # Align dimensions (simple truncation to min length)
    min_len = min(m_p.size(2), m_q.size(2))
    m_p, logs_p = m_p[:, :, :min_len], logs_p[:, :, :min_len]
    m_q, logs_q = m_q[:, :, :min_len], logs_q[:, :, :min_len]

    kl = logs_p - logs_q - 0.5 + 0.5 * ((m_q - m_p) ** 2 + torch.exp(2 * logs_q)) * torch.exp(-2 * logs_p)
    return kl.mean()


if __name__ == "__main__":
    train()
