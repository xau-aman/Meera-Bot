"""Quick test — run MeeraGPT on sample questions."""
import os, sys
import torch
from tokenizers import Tokenizer
from model import MeeraGPT, Config

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TOK_PATH = os.path.join(BASE_DIR, "tokenizer", "meera_tokenizer.json")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = Tokenizer.from_file(TOK_PATH)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
cfg = Config()
for k, v in checkpoint["config"].items():
    setattr(cfg, k, v)
model = MeeraGPT(cfg).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"MeeraGPT loaded ({model.count_params()/1e6:.1f}M params)\n")

questions = [
    "What is an array?",
    "Explain binary search.",
    "Who are you?",
    "What is dynamic programming?",
]

for q in questions:
    prompt = f"<|system|>You are Meera, a smart and slightly witty female AI mentor who helps with coding, DSA, and interviews.<|end|>\n<|user|>{q}<|end|>\n<|meera|>"
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    out = model.generate(idx, max_new_tokens=150, temperature=0.7)
    text = tokenizer.decode(out[0].tolist())
    
    if "<|meera|>" in text:
        response = text.split("<|meera|>")[-1].replace("<|end|>", "").strip()
    else:
        response = text

    print(f"Q: {q}")
    print(f"A: {response[:300]}")
    print("-" * 50)
