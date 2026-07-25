from transformers import AutoProcessor
from pathlib import Path
import json


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_sample(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.loads(f.readline())
    return data


def build_message(record: dict, schema_path: Path) -> list[dict]:
    target_json = json.dumps(record["attributes"])

    schema = json.load(open(schema_path))
    fields = ", ".join(schema["fields"])
    vocab_lines = "\n".join(
        f"- {field}: " + ", ".join(f'"{v}"' for v in schema["vocab"][field])
        for field in schema["fields"]
    )
    system_prompt = (
        "You are a fashion attribute extraction model...\n"
        f"The JSON must contain exactly these keys: {fields}.\n"
        f"Allowed values:\n{vocab_lines}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract the Category, fit, neckline, pattern, length and waistline from the image",
                },
                {"type": "image", "image": record["image"]},
            ],
        },
        {"role": "assistant", "content": target_json},
    ]


if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
    )

    record = load_sample("data/train.jsonl")
    prompt = build_message(record, "data/schema.json")
    print("===== TRAINING FORMAT =====")

    text = processor.apply_chat_template(prompt, tokenize=False)
    print(text)

    print("===== INFERENCE FORMAT =====")
    text = processor.apply_chat_template(
        prompt[:2], tokenize=False, add_generation_prompt=True
    )
    print(text)
