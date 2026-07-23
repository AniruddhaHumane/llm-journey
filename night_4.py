from night_3 import MultiHeadedAttn, softmax
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch


def LayerNorm(X, eps=1e-10):
    x_mean = X.mean(axis=-1, keepdims=True)
    x_std = X.std(axis=-1, keepdims=True)
    return (X - x_mean) / (x_std + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class Linear:
    def __init__(self, d_model, d_ff=256):
        self.linear1 = np.random.rand(d_model, d_ff) / np.sqrt(d_model)
        self.linear2 = np.random.rand(d_ff, d_model) / np.sqrt(d_ff)

    def __call__(self, X):
        layer1 = X @ self.linear1
        gelu_out = gelu(layer1)
        return gelu_out @ self.linear2


class Block:
    def __init__(self, input_shape, d_head, n_heads, d_model) -> None:
        self.mha = MultiHeadedAttn(input_shape, d_head, n_heads, d_model)
        self.linear = Linear(input_shape, d_model)

    def __call__(self, X):
        out1 = X + self.mha(LayerNorm(X))
        out2 = out1 + self.linear(LayerNorm(out1))
        return out2


if __name__ == "__main__":
    gpt_model = AutoModel.from_pretrained("gpt2")
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    vocab_size = 3
    SENTENCE = "Red hat had"
    tokens = gpt_tokenizer.encode(SENTENCE)
    X = gpt_model.get_input_embeddings().weight[tokens].detach().numpy()

    d_model = 768
    d_head = 128
    n_heads = 2
    n_blocks = 2
    input_shape = X.shape[1]

    layers = [Block(input_shape, d_head, n_heads, d_model) for i in range(n_blocks)]
    out = layers[1]((layers[0](X)))
    linear_head = np.random.rand(d_model, vocab_size)
    result = softmax(out @ linear_head)
    print(result, result.shape)
