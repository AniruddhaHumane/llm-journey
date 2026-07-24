"""Minimal decoder-only transformer (PyTorch) — Night 8 training port of transformer.py.
Same structure as the numpy version, now as nn.Modules so autograd + loss.backward() work.
Maps 1:1:
  X @ Wq        -> nn.Linear(..., bias=False)
  wte[ids]      -> nn.Embedding
  layer_norm    -> nn.LayerNorm   (now WITH the learned gamma/beta the numpy version omitted)
  gelu (tanh)   -> nn.GELU(approximate="tanh")
Training runs the whole (B, T) sequence at once with a causal mask (teacher forcing).
The KV cache (Night 6) is an INFERENCE optimization and is intentionally omitted here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnHead(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_head, bias=False)  # was: X @ Wq
        self.Wk = nn.Linear(d_model, d_head, bias=False)
        self.Wv = nn.Linear(d_model, d_head, bias=False)
        self.d_head = d_head

    def forward(self, x):  # x: (B, T, d_model)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)  # each (B, T, d_head)
        scores = Q @ K.transpose(-2, -1) / (self.d_head**0.5)  # (B, T, T), scale by sqrt(d_head)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))  # causal mask (-inf)
        attn = F.softmax(scores, dim=-1)
        return attn @ V  # (B, T, d_head)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.heads = nn.ModuleList([AttnHead(d_model, d_model // n_heads) for _ in range(n_heads)])
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        heads = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, d_model)
        return self.Wo(heads)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # learned gamma/beta now
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))  # sub-layer 1: pre-norm + residual
        x = x + self.mlp(self.ln2(x))  # sub-layer 2: pre-norm + residual
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_blocks, context, d_ff=None):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)  # token embeddings
        self.wpe = nn.Embedding(context, d_model)  # positional embeddings
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d_model)  # final norm
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying (same table both ends)

    def forward(self, idx):  # idx: (B, T) token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.wte(idx) + self.wpe(pos)  # (B, T, d_model)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.lm_head(h)  # (B, T, vocab) logits


if __name__ == "__main__":  # smoke test: shapes only (training loop is your Night 8 exercise)
    torch.manual_seed(0)
    m = GPT(vocab=50, d_model=32, n_heads=4, n_blocks=2, context=64)
    idx = torch.randint(0, 50, (2, 10))  # batch of 2 sequences, length 10
    logits = m(idx)
    print("logits:", tuple(logits.shape), "= (B, T, vocab)")
    print("params:", sum(p.numel() for p in m.parameters()))
