from torch_transformer import GPT
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import SGD, Adam
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import json


def plot_loss_curves(results: dict[str, list[float]], filename: str = "realtime_losses.html"):
    fig = go.Figure()

    for lr_label, losses in results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(losses) + 1)),
                y=losses,
                mode="lines",
                name=f"LR = {lr_label}",
            )
        )

    fig.update_layout(
        title="Training Loss Comparison Across Learning Rates",
        xaxis_title="Step",
        yaxis_title="Cross-Entropy Loss",
        template="plotly_white",
        legend_title="Learning Rate",
    )

    # 1. Convert plot to HTML string
    html_content = fig.to_html(include_plotlyjs="cdn")

    # 3. Write directly to disk (non-blocking, no server needed)
    with open(filename, "w") as f:
        f.write(html_content)


def read_data(path: Path):
    with open(path, "r") as f:
        txt = f.read()
    return txt


def train_local_GPT(
    model_weights, tokens_tensor, lr, epoches, B, T, vocab, rank, alpha, device, verbose=False
):
    m = GPT(
        vocab=vocab, d_model=128, n_heads=4, n_blocks=2, context=64, lora_rank=rank, alpha=alpha
    )
    if model_weights:
        m.load_state_dict(model_weights, strict=False if rank > 0 else True)

    m.to(device)

    # 2. Unfreeze ONLY LoRA parameters (if rank > 0)
    if rank > 0:
        for name, param in m.named_parameters():
            param.requires_grad = False
            if "LoRA" in name:
                param.requires_grad = True

    trainable_params = [p for p in m.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr)
    training_loss_history = []
    validation_loss_history = []
    N = len(tokens_tensor)
    tokens_per_batch = B * T
    num_batches = (N - 1) // tokens_per_batch
    training_batches = int(num_batches * 0.8)
    validation_batches = num_batches - training_batches

    for e in range(epoches):
        epoch_train_loss = []
        epoch_val_loss = []
        for b in range(training_batches):
            start = b * tokens_per_batch
            x = tokens_tensor[start : start + tokens_per_batch].view(B, T)
            y = tokens_tensor[start + 1 : start + tokens_per_batch + 1].view(B, T)
            perm = torch.randperm(x.size(0), device=x.device)
            x = x[perm]
            x = y[perm]
            optimizer.zero_grad()
            y_pred = m(x)
            loss = F.cross_entropy(y_pred.permute(0, 2, 1), y)
            epoch_train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            if verbose:
                print(f"loss per batch {b}: {training_loss_history[-1]}")
        for b in range(validation_batches):
            with torch.inference_mode():
                start = (training_batches + b) * tokens_per_batch
                x = tokens_tensor[start : start + tokens_per_batch].view(B, T)
                y = tokens_tensor[start + 1 : start + tokens_per_batch + 1].view(B, T)
                y_pred = m(x)
                loss = F.cross_entropy(y_pred.permute(0, 2, 1), y)
                epoch_val_loss.append(loss.item())
        training_window_loss = np.array(epoch_train_loss).mean()
        validation_window_loss = np.array(epoch_val_loss).mean()
        training_loss_history.append(training_window_loss)
        validation_loss_history.append(validation_window_loss)
        print(f"Loss {e}: {training_window_loss:.6f}/{validation_window_loss:.6f}")
        if verbose:
            with torch.no_grad():
                greedy_preds = torch.argmax(y_pred[1], dim=-1)
                result = tokenizer.decode(greedy_preds)
                print(f"Sample at epoch={e}: {result}")
    torch.save(m.state_dict(), f"base_model_LoRA_{rank}.pt")
    return training_loss_history, validation_loss_history


if __name__ == "__main__":
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab = 50257
    B = 32
    T = 10
    epoches = 50
    device = "cuda:0"

    # txt = read_data(Path("./shakespeare.txt"))
    txt = read_data(Path("./torch_transformer.py"))  # fine tuning on this for Lora
    tokens = tokenizer(txt, verbose=False)["input_ids"]
    tokens_tensor = torch.tensor(tokens, dtype=torch.long).to(device)
    model_weights = torch.load("./base_model.pt")
    results = {}
    for lr in [
        1e-3
    ]:  # 1e-4, 1e-2, 1e-6]: (good, overfitting and underfitting (without LORA)) # for lora higher works better
        for rank in [1, 8, 32, 128]:
            for alpha in (2, 8, 16):
                print(f"training at LR={lr} and rank={rank}")
                training_loss_history, validation_loss_history = train_local_GPT(
                    model_weights,
                    tokens_tensor,
                    lr,
                    epoches,
                    B,
                    T,
                    vocab,
                    rank,
                    alpha,
                    device=device,
                    verbose=False,
                )
                results[f"train_lr{str(lr)}_r{rank}_a{alpha}"] = training_loss_history
                results[f"val_lr{str(lr)}_r{rank}_a{alpha}"] = validation_loss_history

    plot_loss_curves(results)
