"""Minimal decoder-only transformer (numpy) — teaching reference.
Incorporates every fix from Nights 2-4: stable per-row softmax, causal mask (-inf),
per-head weights, sqrt(d_head) scaling, canonical (d_model, n_heads) MHA API,
pre-norm + residual blocks, 4x FFN, positional embeddings, weight-tied LM head.
NOTE: numpy has no autograd -> forward pass only. Port to torch at Night 8 (training).
"""

import numpy as np


# ---------- helpers ----------
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)  # stability
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x, eps=1e-5):  # simplified: no learned gamma/beta
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ---------- attention ----------
class AttnHead:
    def __init__(self, d_model, d_head):
        s = 1 / np.sqrt(d_model)
        self.Wq = np.random.randn(d_model, d_head) * s
        self.Wk = np.random.randn(d_model, d_head) * s
        self.Wv = np.random.randn(d_model, d_head) * s

    def __call__(self, X):  # X: (T, d_model)
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv  # each (T, d_head)
        scores = Q @ K.T / np.sqrt(Q.shape[1])  # (T, T), scale by sqrt(d_head)
        mask = np.triu(np.ones_like(scores, bool), k=1)  # True above diagonal = future
        scores = np.where(mask, -np.inf, scores)  # causal mask
        return softmax(scores) @ V  # (T, d_head)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.heads = [AttnHead(d_model, d_model // n_heads) for _ in range(n_heads)]
        self.Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    def __call__(self, X):
        return np.concatenate([h(X) for h in self.heads], axis=-1) @ self.Wo  # (T, d_model)


# ---------- feed-forward ----------
class MLP:
    def __init__(self, d_model, d_ff=None):
        d_ff = d_ff or 4 * d_model
        self.W1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.W2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)

    def __call__(self, X):
        return gelu(X @ self.W1) @ self.W2


# ---------- block ----------
class Block:
    def __init__(self, d_model, n_heads, d_ff=None):
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.mlp = MLP(d_model, d_ff)

    def __call__(self, X):
        X = X + self.mha(layer_norm(X))  # sub-layer 1: attention
        X = X + self.mlp(layer_norm(X))  # sub-layer 2: MLP
        return X


# ---------- full model (optional scaffolding) ----------
class GPT:
    def __init__(self, vocab, d_model, n_heads, n_blocks, context, d_ff=None):
        self.wte = np.random.randn(vocab, d_model) * 0.02  # token embeddings
        self.wpe = np.random.randn(context, d_model) * 0.02  # positional embeddings
        self.blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_blocks)]

    def __call__(self, token_ids):  # token_ids: list[int], length T
        T = len(token_ids)
        h = self.wte[token_ids] + self.wpe[:T]  # (T, d_model)  positions injected here
        for blk in self.blocks:
            h = blk(h)
        h = layer_norm(h)  # final norm
        return h @ self.wte.T  # LM head, weight-tied -> logits (T, vocab)
