from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import json
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
import time

model_id = "Qwen/Qwen3-VL-2B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_id)

with open("./data/schema.json", "r") as f:
    schema = json.load(f)

vocab = "\n".join(
    [
        f"- {schema['fields'][i]}: {' | '.join(schema['vocab'][schema['fields'][i]])}"
        for i in range(len(schema["fields"]))
    ]
)


SYSTEM_PROMPT = f"""
You are an expert fashion cataloging AI specialized in garment analysis and attribute extraction. 
Your task is to analyze fashion images and return precise, structured descriptions following standard retail metadata schemas.

Always format your output as valid JSON containing the following fields:
    {vocab}

"""

USER_PROMPT = f"""Analyze this fashion image and extract the garment details. Return a JSON object containing {"', '".join(schema["fields"])}"""

images = list(Path("./data/images/test").glob("*.jpg"))


def apply_chat_template(message: dict):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": message["image"]},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": {"type": "text", "text": message["attributes"]},
        },
    ]


df = pd.read_json("./data/train.jsonl", lines=True).to_dict(orient="records")
data = list(map(apply_chat_template, df))

prompt = processor.apply_chat_template(
    data, tokenize=False, add_generation_prompt=False
)


# 3. Process both text and images
inputs = processor(
    text=[prompt[0]],
    images=[Image.open(data[0][1]["content"][0]["image"])],
    return_tensors="pt",
)

input_ids = inputs["input_ids"][0]  # Shape: [Sequence_Length]

# 4. Fetch the special image token ID from the model or processor
image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

# 5. Calculate token counts using boolean masking
is_image = input_ids == image_token_id
is_text = ~is_image

image_token_count = is_image.sum().item()
text_token_count = is_text.sum().item()
total_token_count = len(input_ids)

print(f"Total Sequence Length : {total_token_count}")
print(f"Image Tokens          : {image_token_count}")
print(f"Text Tokens           : {text_token_count}")

# -------------------------------------------------------------------------
config = model.config
num_layers = getattr(config, "num_hidden_layers", 28)
hidden_size = getattr(config, "hidden_size", 2048)

bytes_per_param = 2  # bfloat16 = 2 bytes
bytes_per_token = 2 * num_layers * hidden_size * bytes_per_param

vram_image_mb = bytes_per_token * image_token_count / (1024**2)
vram_text_mb = bytes_per_token * text_token_count / (1024**2)

#  prefill time contribution = n**2
image_prefill_ratio = (image_token_count**2) / (total_token_count**2)
text_prefill_ratio = (text_token_count**2) / (total_token_count**2)

print("\n--- PREFILL VRAM (KV CACHE) CONTRIBUTION ---")
print(
    f"Image Tokens KV Cache : {vram_image_mb:.2f} MB ({image_token_count / total_token_count:.1%})"
)
print(
    f"Text Tokens KV Cache  : {vram_text_mb:.2f} MB ({text_token_count / total_token_count:.1%})"
)

print("\n--- ATTENTION COMPUTATION CONTRIBUTION O(N^2) ---")
print(f"Image Token Prefill Share : {image_prefill_ratio:.1%}")
print(f"Text Token Prefill Share  : {text_prefill_ratio:.1%}")


def measure_prefill_time(model_inputs):
    # warmup
    with torch.inference_mode():
        _ = model.generate(**model_inputs, max_new_tokens=1)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        _ = model.generate(**model_inputs, max_new_tokens=1)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


# 1. Measure Combined (Image + Text) Prefill
inputs_full = inputs.to("cuda")
time_full = measure_prefill_time(inputs_full)

# 2. Measure Text-Only Prefill
text_ids_only = input_ids[input_ids != image_token_id].unsqueeze(0).to("cuda")
inputs_text_only = {"input_ids": text_ids_only}
time_text_only = measure_prefill_time(inputs_text_only)

# 3. Estimate Image Contribution
time_image_contribution = max(0, time_full - time_text_only)

print("\n--- EMPIRICAL PREFILL TIME (TTFT) ---")
print(f"Total Prefill Time      : {time_full:.2f} ms")
print(f"Text-Only Prefill Time  : {time_text_only:.2f} ms")
print(
    f"Image-Token Contribution: ~{time_image_contribution:.2f} ms ({time_image_contribution / time_full:.1%})"
)
