"""Minimal decoder-only transformer (PyTorch) — Night 8 training port of transformer.py.
GPU-ready version supporting device placement (e.g. model.to(device)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnHead(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_head, bias=False)
        self.Wk = nn.Linear(d_model, d_head, bias=False)
        self.Wv = nn.Linear(d_model, d_head, bias=False)
        self.d_head = d_head

    def forward(self, x):  # x: (B, T, d_model)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)  # each (B, T, d_head)
        scores = Q @ K.transpose(-2, -1) / (self.d_head**0.5)  # (B, T, T)

        T = x.size(1)
        # GPU FIX: explicitly create the mask on x.device
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return attn @ V  # (B, T, d_head)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, lora_rank=8, alpha=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.heads = nn.ModuleList([AttnHead(d_model, d_model // n_heads) for _ in range(n_heads)])
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # LoRA Configuration
        self.lora_rank = lora_rank
        if self.lora_rank > 0:
            self.alpha = alpha
            self.a_LoRA = nn.Linear(self.Wo.in_features, self.lora_rank, bias=False)
            self.b_LoRA = nn.Linear(self.lora_rank, self.Wo.out_features, bias=False)
            # Initialize b_LoRA to zero so initial output matches base Wo exactly
            nn.init.zeros_(self.b_LoRA.weight)

    def forward(self, x):
        heads = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, d_model)

        # Compute base path + LoRA path
        base_out = self.Wo(heads)
        if self.lora_rank > 0:
            lora_out = self.b_LoRA(self.a_LoRA(heads)) * (self.alpha / self.lora_rank)
            return base_out + lora_out
        return base_out


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
    def __init__(self, d_model, n_heads, lora_rank=8, alpha=8, d_ff=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, lora_rank=lora_rank, alpha=alpha)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_blocks, context, lora_rank=8, alpha=8, d_ff=None):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.wpe = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, lora_rank, alpha, d_ff) for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.wte.weight  # Weight tying

    def forward(self, idx):  # idx: (B, T)
        B, T = idx.shape
        # GPU FIX: pos tensor created directly on the input tensor's device
        pos = torch.arange(T, device=idx.device)
        h = self.wte(idx) + self.wpe(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.lm_head(h)


if __name__ == "__main__":
    # Device selection: automatically picks CUDA (NVIDIA), MPS (Apple Silicon), or CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    torch.manual_seed(0)

    # 1. Instantiate Model and move to target device
    model = GPT(vocab=50, d_model=32, n_heads=4, n_blocks=2, context=64, lora_rank=8, alpha=16).to(
        device
    )

    # 2. Create Input Tensor and move to target device
    idx = torch.randint(0, 50, (2, 10), device=device)

    # 3. Forward Pass
    logits = model(idx)

    print("logits shape:", tuple(logits.shape), "= (B, T, vocab)")
    print("logits device:", logits.device)
    print("total params:", sum(p.numel() for p in model.parameters()))
