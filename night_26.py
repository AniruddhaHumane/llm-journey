import torch
import json
import re
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from rich.console import Console
from rich.table import Table
from collections import Counter
from tqdm.auto import tqdm

console = Console()

with open("./data/schema.json", "r") as f:
    schema = json.load(f)

vocab_map = {}
for i in range(len(schema["fields"])):
    vocab_map[schema["fields"][i]] = set(schema["vocab"][schema["fields"][i]])


def print_rich_eval_results(
    y_true_jsons: list[dict],
    y_pred_jsons: list[dict],
):
    """Prints a detailed sample-by-sample GT vs Pred rich table and a per-field metric summary."""

    fields = list(vocab_map.keys())

    # =========================================================================
    # 1. Sample-by-Sample Comparison Table
    # =========================================================================
    comparison_table = Table(
        title="[bold cyan]Detailed Batch Predictions vs. Ground Truth[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
        expand=True,
        min_width=16,
    )

    comparison_table.add_column("Sample #", justify="center", style="dim", width=2)

    for field in fields:
        # Each field gets GT and Pred sub-columns
        comparison_table.add_column(f"{field}\n(GT)", justify="left")
        comparison_table.add_column(f"{field}\n(Pred)", justify="left")

    for i, (gt, pred) in enumerate(zip(y_true_jsons, y_pred_jsons)):
        row_cells = [f"[bold]{i + 1}[/bold]"]

        for field in fields:
            gt_val = str(gt.get(field, "N/A"))
            pred_val = str(pred.get(field, "N/A"))

            # Check if prediction is valid vocab
            valid_options = [str(v) for v in vocab_map[field]]
            is_valid_vocab = pred_val in valid_options

            # Color coding logic
            if gt_val == pred_val:
                # Correct prediction -> Green
                pred_cell = f"[green]✓ {pred_val}[/green]"
            elif not is_valid_vocab:
                # Invalid vocab / Out of domain -> Red + Alert icon
                pred_cell = f"[bold red]⚠ {pred_val}[/bold red]"
            else:
                # Incorrect value -> Yellow/Red
                pred_cell = f"[red]✗ {pred_val}[/red]"

            row_cells.append(f"[dim]{gt_val}[/dim]")
            row_cells.append(pred_cell)

        comparison_table.add_row(*row_cells)

    console.print(comparison_table)
    console.print("\n")


# Standard helper to clean JSON fences
def clean_json(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def compute_vocab_metrics(
    y_true_jsons: list[dict],
    y_pred_jsons: list[dict],
):
    """Calculates accuracy, precision, recall, and F1 per vocabulary field with type normalization."""
    field_true = defaultdict(list)
    field_pred = defaultdict(list)

    # 1. Align predictions and ground truth field-by-field with lowercase string normalization
    for true_item, pred_item in zip(y_true_jsons, y_pred_jsons):
        for field in vocab_map.keys():
            t_val = str(true_item.get(field, "MISSING")).strip().lower()
            p_val = str(pred_item.get(field, "INVALID_VALUE")).strip().lower()

            field_true[field].append(t_val)
            field_pred[field].append(p_val)

    metrics_summary = {}

    for field, raw_valid_vocab in vocab_map.items():
        trues = field_true[field]
        preds = field_pred[field]

        # Field-level accuracy
        acc = accuracy_score(trues, preds)

        # Normalize valid vocabulary to strings and lowercase to match trues/preds
        valid_vocab_str = [str(v).strip().lower() for v in raw_valid_vocab]

        # Detailed precision, recall, F1 evaluated against valid vocabulary
        report = classification_report(
            trues,
            preds,
            labels=valid_vocab_str,
            output_dict=True,
            zero_division=0,
        )

        # Map metrics directly to standard keys expected by print_eval_gate_metrics_table
        metrics_summary[field] = {
            "accuracy": round(acc, 4),
            "precision": round(report["macro avg"]["precision"], 4),
            "recall": round(report["macro avg"]["recall"], 4),
            "f1": round(report["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
            "per_class_breakdown": report,
        }

    return metrics_summary


def eval_gate(y_true: str, y_pred: str, batch_size=8):
    invalid_json_indexes = set()
    y_pred_jsons = []

    # check for invalid JSONs
    for y in range(len(y_pred)):
        try:
            y_pred_jsons.append(json.loads(clean_json(y_pred[y])))
        except Exception:
            invalid_json_indexes.add(y)
    print(f"invalid JSONs generated : {len(invalid_json_indexes)}")

    # Ignore corresponding JSONs from evaluation
    y_true_jsons = [
        json.loads(y_true[y])
        for y in range(len(y_true))
        if y not in invalid_json_indexes
    ]

    # per field values validation
    error_rate_per_vocab = {k: 0 for k in vocab_map.keys()}
    incorrect_values = []
    null_rate = {k: 0 for k in vocab_map.keys()}

    for i, y in enumerate(y_pred_jsons):
        for k, v in y.items():
            if v not in vocab_map[k]:
                error_rate_per_vocab[k] += 1
                incorrect_values.append({f"{i}_{k}": v})
            if v is None:
                null_rate[k] += 1
    print("Error rate per vocab:")
    print(json.dumps(error_rate_per_vocab, indent=2))
    print("% error rate")
    print(
        json.dumps(
            {k: x / batch_size * 100 for k, x in error_rate_per_vocab.items()}, indent=2
        )
    )
    # print(incorrect_values)
    print("null rate:")
    print(json.dumps(null_rate, indent=2))
    # very interesting result, e.g. pred = below-the-knee true = below the knee but y_true sample for above-the-knee has dashes but below the knee does not
    # this is easily fixable by following a fixed format.
    # get per vocab metrics
    metrics = compute_vocab_metrics(y_true_jsons, y_pred_jsons)
    # print_rich_eval_results(y_true_jsons, y_pred_jsons)
    return metrics


def print_eval_gate_metrics_table(metrics: dict):
    """Prints output metrics from eval_gate in a clean Rich table."""

    table = Table(
        title="[bold cyan]Eval Gate Metrics Summary[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
        border_style="bright_black",
        expand=False,
    )

    table.add_column("Field", style="bold white")
    table.add_column("Accuracy", justify="right", style="bold green")
    table.add_column("Precision", justify="right", style="cyan")
    table.add_column("Recall", justify="right", style="yellow")
    table.add_column("F1-Score", justify="right", style="bold blue")

    for field, field_metrics in metrics.items():
        acc = field_metrics.get("accuracy", 0.0)
        prec = field_metrics.get("precision", 0.0)
        rec = field_metrics.get("recall", 0.0)
        f1 = field_metrics.get("f1", 0.0)

        acc_str = (
            f"{acc * 100:.2f}%" if isinstance(acc, float) and acc <= 1.0 else f"{acc}%"
        )
        prec_str = (
            f"{prec * 100:.2f}%"
            if isinstance(prec, float) and prec <= 1.0
            else f"{prec}%"
        )
        rec_str = (
            f"{rec * 100:.2f}%" if isinstance(rec, float) and rec <= 1.0 else f"{rec}%"
        )
        f1_str = f"{f1 * 100:.2f}%" if isinstance(f1, float) and f1 <= 1.0 else f"{f1}%"

        table.add_row(field, acc_str, prec_str, rec_str, f1_str)

    console.print("\n")
    console.print(table)


def parse_json_safely(text: str) -> dict:
    """Extracts and parses JSON from response strings safely."""
    try:
        raw = clean_json(text.strip())
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)
    except Exception:
        return {}


def generate_full_dataset(model, processor, val_dataset, batch_size=8):
    """Iterates through the entire val_dataset in batches and returns all y_true and y_pred JSON dicts."""
    all_y_true = []
    all_y_pred = []

    messages_list = val_dataset["messages"]
    total_samples = len(messages_list)

    console.print(
        f"[bold blue]Processing {total_samples} samples in batches of {batch_size}...[/bold blue]"
    )

    for i in tqdm(range(0, total_samples, batch_size), desc="Evaluating"):
        batch = messages_list[i : i + batch_size]

        # 1. Extract Ground Truth strings
        y_true_batch = [
            next(m["content"][0]["text"] for m in sample if m["role"] == "assistant")
            for sample in batch
        ]

        # 2. Extract prompts and images from batch
        user_turns = [
            [m for m in sample if m["role"] != "assistant"] for sample in batch
        ]
        images = [
            next(
                c["image"]
                for m in msgs
                for c in m["content"]
                if c.get("type") == "image"
            )
            for msgs in user_turns
        ]

        # 3. Format text batch
        prompts = [
            processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in user_turns
        ]

        # 4. Prepare inputs
        inputs = processor(
            text=prompts, images=images, padding=True, return_tensors="pt"
        ).to("cuda")

        # 5. Generate
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.1)

        # 6. Direct Batch Decode
        prompt_len = inputs["input_ids"].shape[1]
        y_pred_raw = processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True
        )

        for gt_str, pred_str in zip(y_true_batch, y_pred_raw):
            all_y_true.append(gt_str)
            all_y_pred.append(pred_str)

    return all_y_true, all_y_pred


def compute_majority_class_baselines(
    y_true: list[str], fields: list[str]
) -> dict[str, dict]:
    """Calculates the majority class baseline accuracy for each field on the validation set."""

    baselines = {}
    total_samples = len(y_true)
    jsons = [parse_json_safely(x) for x in y_true]

    if total_samples == 0:
        return {}

    for field in fields:
        # Extract ground truth values (normalized to string)
        gt_values = [str(item.get(field, "MISSING")) for item in jsons]

        # Find the most frequent class and its count
        counts = Counter(gt_values)
        majority_class, majority_count = counts.most_common(1)[0]

        # Baseline accuracy = frequency of majority class / total samples
        baseline_acc = round(majority_count / total_samples, 4)

        baselines[field] = {
            "majority_class": majority_class,
            "majority_count": majority_count,
            "total_samples": total_samples,
            "baseline_accuracy": baseline_acc,
            "class_distribution": dict(counts),
        }

    return baselines


def print_baseline_table(y_true: list[str], fields_or_vocab: list[str] | dict):
    """Computes majority class baselines and prints a styled Rich table."""
    # Handle passing either a list of fields or a vocab_map dictionary
    fields = (
        list(fields_or_vocab.keys())
        if isinstance(fields_or_vocab, dict)
        else fields_or_vocab
    )

    baselines = compute_majority_class_baselines(y_true, fields)

    table = Table(
        title="[bold yellow]Validation Set Majority Class Baseline Metrics[/bold yellow]",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        expand=False,
    )

    table.add_column("Field", style="bold white")
    table.add_column("Majority Class (Mode)", style="green")
    table.add_column("Support (Count / Total)", justify="center", style="dim")
    table.add_column("Baseline Accuracy", justify="right", style="bold yellow")

    for field, res in baselines.items():
        count_str = f"{res['majority_count']} / {res['total_samples']}"
        acc_str = f"{res['baseline_accuracy'] * 100:.2f}%"

        table.add_row(
            field,
            res["majority_class"],
            count_str,
            acc_str,
        )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    from unsloth import FastVisionModel
    from build_dataset import build_dataset, lazy_load_batch

    # Path to your saved checkpoint directory (e.g., latest step/epoch under ./results or adapter directory)
    CHECKPOINT_PATH = "./results/checkpoint-150"  # Update this to your output_dir or specific checkpoint folder

    console.print(
        f"[bold green]Loading checkpoint from: {CHECKPOINT_PATH}[/bold green]"
    )

    # 1. Load model and processor directly from saved checkpoint
    try:
        model, processor = FastVisionModel.from_pretrained(
            CHECKPOINT_PATH,
            max_seq_length=2048,
            load_in_4bit=True,
        )
    except Exception:
        model, processor = FastVisionModel.from_pretrained(
            CHECKPOINT_PATH,
            max_seq_length=2048,
            load_in_4bit=True,
        )

    # 2. Configure for fast inference
    FastVisionModel.for_inference(model)

    # 3. Left padding configuration for batching
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # 4. Load Validation Dataset
    train_dataset, val_dataset, test_dataset = build_dataset(verbose=False)

    train_dataset = train_dataset.with_transform(lazy_load_batch)
    y_true, y_pred = generate_full_dataset(
        model, processor, train_dataset, batch_size=16
    )
    print_baseline_table(y_true, list(vocab_map.keys()))
    print("---------------------------------------------------------------")
    print("------------------------Train Set------------------------------")
    print("---------------------------------------------------------------")
    metrics = eval_gate(y_true, y_pred, batch_size=len(y_pred))
    print_eval_gate_metrics_table(metrics)

    # Val set evaluation
    print("---------------------------------------------------------------")
    print("------------------------Val Set--------------------------------")
    print("---------------------------------------------------------------")
    val_dataset = val_dataset.with_transform(lazy_load_batch)
    y_true, y_pred = generate_full_dataset(model, processor, val_dataset, batch_size=16)
    metrics = eval_gate(y_true, y_pred, batch_size=len(y_pred))
    print_eval_gate_metrics_table(metrics)

    # test set evaluation
    print("---------------------------------------------------------------")
    print("-----------------------Test Set--------------------------------")
    print("---------------------------------------------------------------")
    test_dataset = test_dataset.with_transform(lazy_load_batch)
    y_true, y_pred = generate_full_dataset(
        model, processor, test_dataset, batch_size=16
    )
    metrics = eval_gate(y_true, y_pred, batch_size=len(y_pred))
    print_eval_gate_metrics_table(metrics)
