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
        self.K = None
        self.V = None

    def __call__(self, X, cache=True):  # X: (T, d_model)
        if not cache:
            # prefill phase
            Q = X @ self.Wq  # each (T, d_head)
            self.K = X @ self.Wk
            self.V = X @ self.Wv
        else:
            # generation phase
            Q = X @ self.Wq  # each (1, d_head)
            if self.K is None:
                self.K = X @ self.Wk
                self.V = X @ self.Wv
            else:
                new_token_K = X @ self.Wk
                new_token_V = X @ self.Wv
                self.K = np.vstack([self.K, new_token_K])
                self.V = np.vstack([self.V, new_token_V])

        scores = (
            Q @ self.K.T / np.sqrt(Q.shape[1])
        )  # (T, T), scale by sqrt(d_head) or (1, T), scale by

        if not cache:
            # still prefill phase need token mask
            mask = np.triu(np.ones_like(scores, bool), k=1)  # True above diagonal = future
            scores = np.where(mask, -np.inf, scores)  # causal mask

        return softmax(scores) @ self.V  # (T, d_head)


class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.heads = [AttnHead(d_model, d_model // n_heads) for _ in range(n_heads)]
        self.Wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

    def __call__(self, X, cache=True):
        return np.concatenate([h(X, cache) for h in self.heads], axis=-1) @ self.Wo  # (T, d_model)


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

    def __call__(self, X, cache=True):
        X = X + self.mha(layer_norm(X), cache)  # sub-layer 1: attention
        X = X + self.mlp(layer_norm(X))  # sub-layer 2: MLP
        return X


# ---------- full model (optional scaffolding) ----------
class GPT:
    def __init__(self, vocab, d_model, n_heads, n_blocks, context, d_ff=None):
        self.wte = np.random.randn(vocab, d_model) * 0.02  # token embeddings
        self.wpe = np.random.randn(context, d_model) * 0.02  # positional embeddings
        self.blocks = [Block(d_model, n_heads, d_ff) for _ in range(n_blocks)]

    def __call__(self, token_ids, cache=True, start_pos=0):  # token_ids: list[int], length T
        T = len(token_ids)
        h = (
            self.wte[token_ids] + self.wpe[start_pos : start_pos + T]
        )  # (T, d_model)  positions injected here
        for blk in self.blocks:
            h = blk(h, cache)
        h = layer_norm(h)  # final norm
        return h @ self.wte.T  # LM head, weight-tied -> logits (T, vocab)

    def get_token(self, logits, T=1):
        return np.random.choice(len(logits), p=softmax(logits / T))


# real gpt model
if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    gpt_model = AutoModel.from_pretrained("gpt2")
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    SENTENCE = "RED HAT HAD"
    tokens = gpt_tokenizer.encode(SENTENCE)
    vocab, d_model = gpt_model.get_input_embeddings().weight.shape
    n_heads = 4
    n_blocks = 4
    context = 1024
    d_ff = None
    n = len(tokens)
    model = GPT(vocab, d_model, n_heads, n_blocks, context)
    # Prefill Phase
    new_token_logits = model(tokens, cache=False)[-1]
    pos = len(tokens)
    # Decode Phase
    for i in range(n):
        next_token = model.get_token(new_token_logits)
        new_token_logits = model(np.array([next_token]), cache=True, start_pos=pos)[-1]
        pos += 1
        tokens.append(next_token)
    print(gpt_tokenizer.decode(tokens))

    old_k, old_v = [], []
    for b in model.blocks:
        for h in b.mha.heads:
            old_k.append(h.K)
            old_v.append(h.V)
    new_token_logits = model(tokens, cache=False)
    new_k, new_v = [], []
    for b in model.blocks:
        for h in b.mha.heads:
            new_k.append(h.K)
            new_v.append(h.V)
    old_k = np.concat(old_k)
    old_v = np.concat(old_v)
    new_k = np.concat(new_k)
    new_v = np.concat(new_v)
    assert (abs(old_k - new_k) < 1e-10).all()
    assert (abs(old_v - new_v) < 1e-10).all()
