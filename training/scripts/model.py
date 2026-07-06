"""Meera GPT — small transformer model (~50M params, fits 8GB VRAM)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size: int = 8192
    max_seq_len: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    pad_token_id: int = 0

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each: (B, T, n_heads, head_dim)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, n_heads, T, head_dim)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)  # SwiGLU gate
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class MeeraGPT(nn.Module):
    def __init__(self, cfg: Config = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos_emb = nn.Embedding(self.cfg.max_seq_len, self.cfg.d_model)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.layers = nn.ModuleList([TransformerBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.norm = RMSNorm(self.cfg.d_model)
        self.head = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = self.dropout(tok + pos)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.cfg.pad_token_id)

        return logits, loss

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=256, temperature=0.8, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)

            # Stop at <|end|> token (id=7)
            if next_tok.item() == 7:
                break

        return idx

if __name__ == "__main__":
    cfg = Config()
    model = MeeraGPT(cfg)
    print(f"MeeraGPT — {model.count_params() / 1e6:.1f}M parameters")
    print(f"Config: {cfg.n_layers} layers, {cfg.n_heads} heads, d_model={cfg.d_model}")

    # Test forward pass
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    logits, loss = model(x, x)
    print(f"Test forward pass: logits shape={logits.shape}, loss={loss.item():.4f}")
