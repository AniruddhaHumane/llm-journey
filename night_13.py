import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def read_data(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def analyse_token_length(dataset):
    sample_lengths = [
        len(tokenizer.encode(p + c))
        for p, c in zip(dataset["prompt"], dataset["completion"])
    ]

    # 3. Calculate percentiles
    p50 = np.percentile(sample_lengths, 50)
    p95 = np.percentile(sample_lengths, 95)
    p99 = np.percentile(sample_lengths, 99)
    max_len = max(sample_lengths)

    print(f"Median length (p50): {p50:.0f} tokens")
    print(f"95th Percentile (p95): {p95:.0f} tokens")
    print(f"99th Percentile (p99): {p99:.0f} tokens")
    print(f"Longest sample in dataset: {max_len} tokens")


if __name__ == "__main__":
    data = read_data(Path("./sample_qa_data.json"))
    split = int(len(data) * 0.8)
    train_data, val_data = (
        Dataset.from_list(data[:split]),
        Dataset.from_list(data[split:]),
    )

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    analyse_token_length(train_data)

    # Initialize the collator
    args = SFTConfig(
        output_dir="./results",
        completion_only_loss=False,  # Automatically masks prompt tokens
        max_length=16,  # based on token analysis
        # --- training and validation ---
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        # --- Loss & Metrics Logging ---
        logging_steps=2,  # Log training loss every 10 steps
        eval_strategy="steps",  # Run evaluation periodically
        eval_steps=10,  # Evaluate loss on validation set every 50 steps
        save_strategy="steps",  # Save model checkpoints
        save_steps=50,
    )

    trainer = SFTTrainer(
        model, train_dataset=train_data, eval_dataset=val_data, args=args
    )
    # masking confirmation
    print(f"unmasked sample: {next(iter(trainer.get_train_dataloader()))['labels'][0]}")
    # trainer.train() - no need to train this

    # Initialize the collator
    args = SFTConfig(
        output_dir="./results",
        completion_only_loss=True,  # Automatically masks prompt tokens
        max_length=16,  # based on token analysis
        # --- training and validation ---
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        # --- Loss & Metrics Logging ---
        logging_steps=2,  # Log training loss every 10 steps
        eval_strategy="steps",  # Run evaluation periodically
        eval_steps=10,  # Evaluate loss on validation set every 50 steps
        save_strategy="steps",  # Save model checkpoints
        save_steps=50,
    )

    trainer = SFTTrainer(
        model, train_dataset=train_data, eval_dataset=val_data, args=args
    )
    # masking confirmation
    print(f"\nmasked sample: {next(iter(trainer.get_train_dataloader()))['labels'][0]}")
    trainer.train()

    trainer.model.eval()
    with torch.inference_mode():
        tokens = tokenizer.encode(val_data[0]["prompt"], return_tensors="pt")
        generated_tokens1 = trainer.model.generate(
            tokens,
            max_new_tokens=16,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(generated_tokens1, skip_special_tokens=True)
        print(response)

        tokens = tokenizer.encode(val_data[1]["prompt"], return_tensors="pt")
        generated_tokens1 = trainer.model.generate(
            tokens,
            max_new_tokens=16,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(generated_tokens1, skip_special_tokens=True)
        print(response)
