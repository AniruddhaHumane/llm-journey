import json
from pathlib import Path
import pandas as pd
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich import box
from PIL import Image


with open("./data/schema.json", "r") as f:
    schema = json.load(f)

vocab = "\n".join(
    [
        f"- {schema['fields'][i]}: {' | '.join(schema['vocab'][schema['fields'][i]])}"
        for i in range(len(schema["fields"]))
    ]
)

vocab_map = {}
for i in range(len(schema["fields"])):
    vocab_map[schema["fields"][i]] = set(schema["vocab"][schema["fields"][i]])


SYSTEM_PROMPT = f"""
You are an expert fashion cataloging AI specialized in garment analysis and attribute extraction. 
Your task is to analyze fashion images and return precise, structured descriptions following standard retail metadata schemas.

Always format your output as valid JSON containing the following fields:
    {vocab}

""".strip()

USER_PROMPT = f"""Analyze this fashion image and extract the garment details. Return a JSON object containing '{"', '".join(schema["fields"])}'"""

images = list(Path("./data/images/test").glob("*.jpg"))


def print_stats(path):

    # statistics
    # 2. Extract raw attributes
    raw_data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            raw_data.append(item.get("attributes", {}))

    df = pd.DataFrame(raw_data)

    # 3. Decoding helper
    def decode_attribute(val, vocab_list):
        if pd.isna(val) or val is None:
            return "None (Missing)"
        if isinstance(val, int) and 0 <= val < len(vocab_list):
            return vocab_list[val]
        return str(val)

    # Process columns
    processed_df = pd.DataFrame()
    for field in schema["fields"]:
        if field in df.columns:
            vocab_list = schema["vocab"][field]
            processed_df[field] = df[field].apply(
                lambda x: decode_attribute(x, vocab_list)
            )

    # 4. Generate Rich Tables
    console = Console()
    total_samples = len(processed_df)

    for field in schema["fields"]:
        if field not in processed_df.columns:
            continue

        # Compute value counts and percentages
        counts = processed_df[field].value_counts().reset_index()
        counts.columns = ["class", "count"]
        counts["percentage"] = (counts["count"] / total_samples) * 100

        table = Table(
            title=f"Class Distribution: {field.upper()} (Total N={total_samples})",
            box=box.ROUNDED,
            header_style="bold cyan",
        )

        table.add_column("Class Label", style="bold white")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")

        for _, row in counts.iterrows():
            is_missing = row["class"] == "None (Missing)"
            style = "dim red" if is_missing else None

            table.add_row(
                str(row["class"]),
                f"{row['count']:,}",
                f"{row['percentage']:.2f}%",
                style=style,
            )

        console.print(table)
        console.print()


def _build_dataset(path, verbose=True):
    # 1. Initialize trackers
    null_rate = {attribute: 0 for attribute in schema["fields"]}
    unknown_rate = {attribute: 0 for attribute in schema["fields"]}
    missing_images = []
    # 2. Updated Chat Template Function

    def process_example(message: dict):
        img_path = Path(message["image"])

        # Skip if image does not exist
        if not img_path.exists():
            missing_images.append(message["image"])
            return None

        # Track null attributes & validate vocab
        for attribute, value in message["attributes"].items():
            if value is not None:
                if value not in vocab_map[attribute]:
                    unknown_rate[attribute] += 1
            else:
                null_rate[attribute] += 1

        # Return structure where ALL 'content' fields are lists of dicts
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(message["attributes"]),
                        }
                    ],
                },
            ]
        }

    # 3. Process records & filter Nones
    df = pd.read_json(path, lines=True)
    df_records = df.to_dict(orient="records")
    formatted_list = [
        res for item in df_records if (res := process_example(item)) is not None
    ]

    # 4. Convert directly to Hugging Face Dataset (No PyArrow schema mismatch!)
    dataset = Dataset.from_list(formatted_list)
    if verbose:
        print(f"Null rate for each category: {json.dumps(null_rate, indent=2)}")
        print(f"unknown rate for each category: {json.dumps(unknown_rate, indent=2)}")
        print(f"image doesn't exist count: {missing_images}")
        print_stats(path)
    return dataset


def build_dataset(
    train_path="./data/train.jsonl",
    test_path="./data/test.jsonl",
    split=0.2,
    verbose=True,
):
    train_dataset = _build_dataset(train_path, verbose)
    test_dataset = _build_dataset(test_path, verbose)

    out = train_dataset.train_test_split(test_size=split, seed=42)
    train, val = out["train"], out["test"]
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test_dataset)}")
    if verbose:
        from transformers import AutoProcessor

        model_id = "Qwen/Qwen3-VL-2B-Instruct"
        processor = AutoProcessor.from_pretrained(model_id)

        print(
            processor.apply_chat_template(
                train[0]["messages"], tokenize=False, add_generation_prompt=False
            )
        )
    return train, val, test_dataset


# train, val, test = build_dataset()


# fig.show()
# pattern dominated by plain
# fit is normal like distribtion with majority being regular
# length is again normal like with majority being mini
# category as expected is dress dominant


def lazy_load_batch(batch):
    messages_col = batch["messages"]

    # If first element is a dict (a turn), it's a single conversation (e.g. dataset[0])
    if len(messages_col) > 0 and isinstance(messages_col[0], dict):
        conversations = [messages_col]
        is_single = True
    else:
        conversations = messages_col
        is_single = False

    processed_conversations = []
    for conversation in conversations:
        new_conversation = []
        for turn in conversation:
            new_contents = []
            for item in turn["content"]:
                # Open string path as PIL.Image on the fly
                if item.get("type") == "image" and isinstance(item.get("image"), str):
                    img = Image.open(item["image"]).convert("RGB")
                    new_contents.append({"type": "image", "image": img})
                else:
                    new_contents.append(
                        {k: v for k, v in item.items() if v is not None}
                    )  # ← drop image:None
            new_conversation.append({"role": turn["role"], "content": new_contents})
        processed_conversations.append(new_conversation)

    if is_single:
        return {"messages": processed_conversations[0]}
    return {"messages": processed_conversations}
