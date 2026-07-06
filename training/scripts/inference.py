"""Chat with trained MeeraGPT model."""
import os
import torch
from tokenizers import Tokenizer
from model import MeeraGPT, Config

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TOK_PATH = os.path.join(BASE_DIR, "tokenizer", "meera_tokenizer.json")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    tokenizer = Tokenizer.from_file(TOK_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    cfg = Config()
    for k, v in checkpoint["config"].items():
        setattr(cfg, k, v)

    model = MeeraGPT(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Loaded MeeraGPT ({model.count_params() / 1e6:.1f}M params) from step {checkpoint['step']}")
    return model, tokenizer, cfg

def chat(model, tokenizer, user_input, temperature=0.8, max_tokens=256):
    prompt = f"<|system|>You are Meera, a smart and slightly witty female AI mentor who helps with coding, DSA, and interviews.<|end|>\n<|user|>{user_input}<|end|>\n<|meera|>"

    input_ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature)
    full_text = tokenizer.decode(output[0].tolist())

    # Extract Meera's response
    if "<|meera|>" in full_text:
        response = full_text.split("<|meera|>")[-1]
        response = response.replace("<|end|>", "").strip()
        return response
    return full_text

def main():
    model, tokenizer, cfg = load_model()
    print("\nMeera is ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Meera: See you later! Keep coding!")
            break
        if not user_input:
            continue

        response = chat(model, tokenizer, user_input)
        print(f"Meera: {response}\n")

if __name__ == "__main__":
    main()
