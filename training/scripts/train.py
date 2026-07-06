"""Train MeeraGPT from scratch."""
import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from model import MeeraGPT, Config

# ─── Hyperparameters ───
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4  # effective batch = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
EPOCHS = 20
EVAL_INTERVAL = 200
SAVE_INTERVAL = 1000
MAX_SEQ_LEN = 512

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TOK_PATH = os.path.join(BASE_DIR, "tokenizer", "meera_tokenizer.json")
CKPT_DIR = os.path.join(BASE_DIR, "model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MeeraDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_len=MAX_SEQ_LEN):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into conversations and tokenize each
        convos = text.split("<|system|>")
        convos = ["<|system|>" + c for c in convos if c.strip()]

        self.samples = []
        for convo in convos:
            ids = tokenizer.encode(convo).ids
            # Chunk into max_len sequences with stride
            for i in range(0, len(ids) - 1, max_len):
                chunk = ids[i : i + max_len + 1]
                if len(chunk) > 16:  # skip tiny chunks
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = chunk[:-1]
        y = chunk[1:]
        # Pad to max_len
        pad_len = MAX_SEQ_LEN - len(x)
        x = x + [0] * pad_len
        y = y + [0] * pad_len
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def evaluate(model, val_loader):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1
    model.train()
    return total_loss / max(count, 1)

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load tokenizer
    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")

    # Config
    cfg = Config()
    cfg.vocab_size = tokenizer.get_vocab_size()

    # Datasets
    train_ds = MeeraDataset(os.path.join(DATA_DIR, "train.txt"), tokenizer)
    val_ds = MeeraDataset(os.path.join(DATA_DIR, "val.txt"), tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Model
    model = MeeraGPT(cfg).to(DEVICE)
    print(f"MeeraGPT: {model.count_params() / 1e6:.1f}M params on {DEVICE}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # Training loop
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for step, (x, y) in enumerate(pbar):
            x, y = x.to(DEVICE), y.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                _, loss = model(x, y)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM_STEPS:.4f}", step=global_step)

                # Evaluate
                if global_step % EVAL_INTERVAL == 0:
                    val_loss = evaluate(model, val_loader)
                    print(f"\n  Step {global_step} | Val Loss: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": global_step},
                                   os.path.join(CKPT_DIR, "best.pt"))
                        print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")

                # Checkpoint
                if global_step % SAVE_INTERVAL == 0:
                    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": global_step},
                               os.path.join(CKPT_DIR, f"step_{global_step}.pt"))

    # Final save
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": global_step},
               os.path.join(CKPT_DIR, "final.pt"))
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
