from transformers import AutoModel, AutoTokenizer
import numpy as np

gpt_model = AutoModel.from_pretrained("gpt2")
gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

SENTENCE = "RED HAT HAD"
tokens = gpt_tokenizer.encode(SENTENCE)
X = gpt_model.get_input_embeddings().weight[tokens].detach().numpy()


def softmax(X, axis=-1):
    X_max = X.max(axis=axis).reshape(-1, 1)
    e_x = np.exp(X - X_max)
    return e_x / e_x.sum(axis=axis).reshape(-1, 1)


d_model = 768
d_k = 128
n_heads = 2
input_shape = X.shape[1]


class AttnHead:
    d_head = None

    def __init__(self, d_head):
        self.d_head = d_head
        self.Wq = np.random.rand(input_shape, d_head)
        self.Wk = np.random.rand(input_shape, d_head)
        self.Wv = np.random.rand(input_shape, d_head)

    def __call__(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        proj = Q @ K.T / np.sqrt(self.d_head)
        # print(f"proj shape: {proj.shape}")
        # print(proj)
        mask = np.triu(np.ones_like(proj, bool), k=1)
        proj_masked = np.where(mask, -np.inf, proj)
        # print(f"proj_mask: {proj_masked.shape}")
        # print(proj_masked)
        attn = softmax(proj_masked)
        # print(f"attn: {attn.shape}")
        # print(attn, attn.sum(axis=-1))
        return attn @ V


class MultiHeadedAttn:
    def __init__(self, d_k, n_heads):
        if d_k % n_heads:
            raise ValueError("d_k must be divisible by n_heads")
        self.heads = [AttnHead(d_k // n_heads) for i in range(n_heads)]
        self.W = np.random.rand(d_k, d_model)

    def __call__(self, X):
        values = [head(X) for head in self.heads]
        return np.concat(values, axis=1) @ self.W


mha = MultiHeadedAttn(d_k, n_heads)
v = mha(X)
print(v.shape)
