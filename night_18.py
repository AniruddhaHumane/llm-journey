import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from bitsandbytes.nn import Linear4bit
from peft import get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer


model_id = "Qwen/Qwen3-VL-2B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

allocated_bytes = torch.cuda.memory_allocated()
allocated_gb = allocated_bytes / (1024**3)

model_fp4 = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

newly_allocated_bytes = torch.cuda.memory_allocated()
newly_allocated_gb = newly_allocated_bytes / (1024**3)

print(f"loaded model size in GB: {newly_allocated_gb - allocated_gb}")

# processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

newly_allocated_bytes_16 = torch.cuda.memory_allocated()
newly_allocated_gb_16 = newly_allocated_bytes_16 / (1024**3)
print(f"loaded model16 size in GB: {newly_allocated_gb_16 - newly_allocated_gb}")

linear_4bit_layers = {}

for name, module in model_fp4.named_modules():
    if isinstance(module, Linear4bit):
        linear_4bit_layers[name] = module

for name, module in list(linear_4bit_layers.items())[:2]:
    qs = module.weight.quant_state

    print(f"Layer: {name}")
    print(f"  Shape: {qs.shape} | Quant Type: {qs.quant_type}")
    print(f"  Primary Blocksize: {qs.blocksize}")
    print(f"  Absmax Shape: {qs.absmax.shape}")

    # Check if Double Quantization is active
    if getattr(qs, "nested", False) and qs.state2 is not None:
        print("  [Double Quantization Active]")
        print(f"    Nested Blocksize: {qs.state2.blocksize}")
        print(f"    Nested Absmax Shape: {qs.state2.absmax.shape}")
        print(f"    Nested Quant Map (Code): {qs.state2.code.shape}")
        print(f"    Nested Offset: {qs.offset}")
    else:
        print("  [No Double Quantization]")
    print("-" * 50)


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
)

lora_model = get_peft_model(
    model_fp4,
    lora_config,
    autocast_adapter_dtype=False,
)
lora_model.print_trainable_parameters()


# Extract and display the first 4 LoRA-adapted quantized layers
target_count = 4
found_count = 0

print(f"{'Layer Name':<45} | {'Base Weight':<12} | {'lora_A':<12} | {'lora_B':<12}")
print("-" * 90)

for name, module in lora_model.named_modules():
    # PEFT wraps quantized layers in a LoraLayer container
    if isinstance(module, LoraLayer):
        # Extract base layer weight dtype (bitsandbytes stores 4-bit weights as uint8)
        base_dtype = module.base_layer.weight.quant_type

        # Extract adapter weights (default adapter name is 'default')
        lora_a_dtype = module.lora_A["default"].weight.dtype
        lora_b_dtype = module.lora_B["default"].weight.dtype
        # lora_b_grad_dtype = module.lora_B["default"].weight.grad.dtype # this fails as grad is None

        # Format layer name for clean table display
        short_name = name if len(name) <= 45 else "..." + name[-42:]
        print(
            f"{short_name:<45} | {str(base_dtype):<12} | {str(lora_a_dtype):<12} | {str(lora_b_dtype):<12}"
        )

        found_count += 1
        if found_count >= target_count:
            break
