# Attention dot product form scratch

import numpy as np
import tiktoken
from transformers import AutoModel
import matplotlib.pyplot as plt

np.random.seed(0)

SENTENCE = "red hat looks good"
d_k = 128
enc = tiktoken.encoding_for_model("gpt-2")
model = AutoModel.from_pretrained("gpt2")


def get_tokens(sentence: str):
    tokens = enc.encode(sentence)
    print(tokens)
    return tokens
    # hf_enc = AutoTokenizer.from_pretrained("gpt2")
    # tokens2 = hf_enc.encode(SENTENCE)
    # print(tokens2)
    # tokens and tokens2 are exactly the same


def get_embeddings(tokens):
    embeddings = model.get_input_embeddings().weight[tokens]
    print(embeddings.shape)
    return embeddings


tokens = get_tokens(SENTENCE)
embeddings = get_embeddings(tokens)

X_shape = embeddings.detach().numpy().shape[1]
# implement attn = softmax(QKᵀ/√d)·V
Wq = np.random.rand(X_shape, d_k)
Wk = np.random.rand(X_shape, d_k)
Wv = np.random.rand(X_shape, d_k)


def softmax(arr: np.array, axis=-1):
    max_axis = np.max(arr, axis=-1).reshape(-1, 1)
    e_x = np.exp(arr - max_axis)
    return e_x / np.sum(e_x, axis=-1).reshape(-1, 1)


def project_qkv(embeddings, d_k=128):
    X = embeddings.detach().numpy()
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    return Q, K, V


Q, K, V = project_qkv(embeddings)


def plot_mat(matrix: np.array, title=None):
    fig, ax = plt.subplots()
    # 2. Plot the matrix as an image
    _ = ax.imshow(matrix, cmap="Blues")
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            # Determine text color based on background intensity for readability
            text_color = "white" if matrix[i, j] > 0.60 else "black"
            # ax.text(x, y, string, horizontalalignment, verticalalignment)
            ax.text(
                j,
                i,
                str(matrix[i, j].round(3)),
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )
    plt.title(title)
    plt.show()


# Attention Without dividing by √d
attn = softmax((Q @ K.T))
out = attn @ V
print((Q @ K.T).std(), (Q @ K.T).mean(), (Q @ K.T).max())
plot_mat(attn, "attention without dividing by √d")

# Attention after dividing by √d
attn = softmax((Q @ K.T) / np.sqrt(d_k))
out = attn @ V
plot_mat(attn, "attention after dividing by √d")

# change one token:
SENTENCE = "red rat looks good"
tokens = get_tokens(SENTENCE)
embeddings = get_embeddings(tokens)
Q, K, V = project_qkv(embeddings)

attn2 = softmax((Q @ K.T) / np.sqrt(d_k))
out2 = attn2 @ V
plot_mat(attn2, "attention after changing one token")
