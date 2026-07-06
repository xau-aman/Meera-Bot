"""Train a BPE tokenizer on Meera's corpus."""
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tokenizer")

SPECIAL_TOKENS = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|meera|>", "<|end|>"]
VOCAB_SIZE = 8192  # Small vocab for small model

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    files = [os.path.join(DATA_DIR, f) for f in ["train.txt", "val.txt"]]
    tokenizer.train(files, trainer)

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    out_path = os.path.join(OUT_DIR, "meera_tokenizer.json")
    tokenizer.save(out_path)

    # Test
    test_texts = [
        "<|system|>You are Meera.<|end|>",
        "<|user|>What is an array?<|end|>",
        "<|meera|>An array is a contiguous block of memory.<|end|>",
    ]
    for t in test_texts:
        enc = tokenizer.encode(t)
        print(f"  '{t[:50]}...' -> {len(enc.ids)} tokens")

    print(f"\nTokenizer saved to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    main()
