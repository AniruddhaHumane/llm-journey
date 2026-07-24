from torch_transformer import GPT
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.optim import Adam
from pathlib import Path
import numpy as np


def read_data(path: Path):
    with open(path, "r") as f:
        txt = f.read()
    return txt


if __name__ == "__main__":
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab = 50257
    steps = 500
    B = 32
    T = 10

    m = GPT(vocab=vocab, d_model=128, n_heads=4, n_blocks=2, context=64)
    txt = read_data(Path("./shakespeare.txt"))  # keeping it short < 1000
    # otherwise I get this error: [transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (1236 > 1024). Running this sequence through the model will result in indexing errors

    optimizer = Adam(m.parameters())
    loss_history = []

    tokens = tokenizer(txt, verbose=False)["input_ids"]
    tokens_tensor = torch.tensor(tokens, dtype=torch.long)

    # 2. Inside your step loop:
    for step in range(steps):
        # Sample B random starting indices directlys
        ix = torch.randint(len(tokens_tensor) - T - 1, (B,))

        # Grab B slices of length T instantly on GPU/CPU
        x = torch.stack([tokens_tensor[i : i + T] for i in ix])
        y = torch.stack([tokens_tensor[i + 1 : i + T + 1] for i in ix])

        optimizer.zero_grad()
        y_pred = m(x)
        loss = F.cross_entropy(y_pred.permute(0, 2, 1), y)
        loss_history.append(loss.item())
        loss.backward()
        optimizer.step()
        if (step + 1) % 10 == 0:
            print(f"average loss per 10 steps: {np.array(loss_history).mean()}")
            loss_history = []
        if (step + 1) % 50 == 0:
            with torch.no_grad():
                greedy_preds = torch.argmax(y_pred[0], dim=-1)
                result = tokenizer.decode(greedy_preds)
                print(f"Sample at step={step}: {result}")
