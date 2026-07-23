from local_transformer import GPT, softmax
import numpy as np

if __name__ == "__main__":
    vocab = 50000
    d_model = 128
    n_heads = 4
    n_blocks = 4
    context = 1024
    d_ff = None

    model = GPT(vocab, d_model, n_heads, n_blocks, context, d_ff)
    tokens = [1, 10, 100, 1000, 1000]
    new_token = model(tokens)[-1]
    # Greedy:
    print(f"maximum value of token {np.argmax(new_token)}")
    # Softmax sampling
    print(f"softmax with max: {np.random.choice(len(new_token), p=softmax(new_token))}")
    # Temperature
    T = 1
    print(f"T = 1: {np.random.choice(len(new_token), p=softmax(new_token / T))}")
    T = 0.3
    print(f"T < 1: {np.random.choice(len(new_token), p=softmax(new_token / T))}")
    T = 4
    print(f"T > 1: {np.random.choice(len(new_token), p=softmax(new_token / T))}")
    # Top-k
    k = 10
    top_k_idxs = np.argpartition(new_token, -k)[-k:]
    out_idx = np.random.choice(len(top_k_idxs), p=softmax(new_token[top_k_idxs]))
    print(f"top-k: {top_k_idxs[out_idx]}")
    # top-p
    p = 0.5
    order = np.argsort(new_token)[::-1]
    cum = np.cumsum(softmax(new_token[order]))
    keep = order[: np.searchsorted(cum, p) + 1]  # +1 includes the crossing token; never empty
    token = keep[np.random.choice(len(keep), p=softmax(new_token[keep]))]
    print(f"top-p: {token}")
