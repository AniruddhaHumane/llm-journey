from peft import get_peft_model, LoraConfig
from transformers import Qwen3VLForConditionalGeneration
import torch

model_id = "Qwen/Qwen3-VL-2B-Instruct"

# Already done so commenting it out...
# base_model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,  # or torch.float16
#     device_map="auto",  # Load on CPU if VRAM is tight, or "auto" for GPU
# )
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     task_type="CAUSAL_LM",
#     lora_dropout=0.05,
#     bias="none",
# )

# lora_model = get_peft_model(
#     base_model,
#     lora_config,
#     autocast_adapter_dtype=False,
# )

# lora_model.train()

# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, lora_model.parameters()), lr=1e-4
# )

# input_ids = torch.randint(0, 1000, (1, 8)).to(lora_model.device)
# labels = input_ids.clone() + 1

# # sample forward loops
# for i in range(10):
#     out = lora_model(input_ids=input_ids, labels=labels)
#     loss = out.loss

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# model = lora_model.merge_and_unload()
# model.save_pretrained("./models/qwen_lora")

# lora does not exist here and it is a plain pytorch model
# print(
#     sum([("lora" in name or "base_layer" in name) for name, _ in model.named_modules()])
# )

# PTQ
from transformers import AutoProcessor
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MERGED = "./models/qwen_lora"  # ← what you saved after merge_and_unload
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MERGED, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# --- CALIBRATION DATA (this is the star of the night) ---
# a few hundred short, representative samples; tokenize them
ds = (
    load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    .shuffle(seed=42)
    .select(range(256))
)


def preprocess_text_for_vlm(example):
    # Convert text strings to the multimodal content dict structure
    formatted_messages = []
    for msg in example["messages"]:
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        formatted_messages.append({"role": msg["role"], "content": content})

    # Apply the VLM chat template (formats special tokens like <|im_start|>)
    rendered_text = processor.apply_chat_template(
        formatted_messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize using the internal tokenizer of the processor
    encodings = processor.tokenizer(
        rendered_text,
        truncation=True,
        max_length=512,
        padding=False,
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }


ds = ds.map(preprocess_text_for_vlm, remove_columns=ds.column_names)

# --- THE RECIPE: GPTQ, 4-bit weights / 16-bit activations, skip the output head ---
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    processor=processor,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=256,
)

model.save_pretrained("./models/merged-gptq-w4a16")
processor.save_pretrained("./models/merged-gptq-w4a16")

#### BENCHMARK
import time


def print_model_vram(model, model_name="Model"):
    param_size = 0
    buffer_size = 0

    # 1. Calculate weights size
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    # 2. Calculate registered buffers size (e.g. KV cache, position IDs)
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_bytes = param_size + buffer_size
    total_mb = total_bytes / (1024**2)
    total_gb = total_bytes / (1024**3)

    print(f"=== {model_name} Memory Usage ===")
    print(f"Parameters : {param_size / (1024**2):.2f} MB")
    print(f"Buffers    : {buffer_size / (1024**2):.2f} MB")
    print(f"Total VRAM : {total_mb:.2f} MB ({total_gb:.2f} GB)")


QUANTIZED = "./models/merged-gptq-w4a16"  # ← what you saved after merge_and_unload
model = Qwen3VLForConditionalGeneration.from_pretrained(
    QUANTIZED, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(QUANTIZED)


def toks_per_sec(model, processor, prompt, n_new=128):
    # Specify text= explicitly so processor doesn't treat it as an image
    ids = processor(text=prompt, return_tensors="pt").to(model.device)

    # WARMUP
    model.generate(**ids, max_new_tokens=8)
    torch.cuda.synchronize()

    t = time.time()
    out = model.generate(**ids, max_new_tokens=n_new, do_sample=False)
    torch.cuda.synchronize()

    gen = out.shape[-1] - ids.input_ids.shape[-1]
    return gen / (time.time() - t)


print(toks_per_sec(model, processor, "Capital of france is "))
print_model_vram(model, "Qwen3-VL (4-bit GPTQ)")


MERGED = "./models/qwen_lora"  # ← what you saved after merge_and_unload
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MERGED, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")


def toks_per_sec(model, processor, prompt, n_new=128):
    # Specify text= explicitly so processor doesn't treat it as an image
    ids = processor(text=prompt, return_tensors="pt").to(model.device)

    # WARMUP
    model.generate(**ids, max_new_tokens=8)
    torch.cuda.synchronize()

    t = time.time()
    out = model.generate(**ids, max_new_tokens=n_new, do_sample=False)
    torch.cuda.synchronize()

    gen = out.shape[-1] - ids.input_ids.shape[-1]
    return gen / (time.time() - t)


print(toks_per_sec(model, processor, "Capital of france is "))
print_model_vram(model, "Qwen3-VL (QLoRA)")
