from unsloth import FastVisionModel
import os

os.environ["UNSLOTH_COMPILE_LOCATION"] = "0"  # Disables dynamic module compilation

"""
env setup
uv add torch torchvision bitsandbytes llmcompressor trl unsloth torchao numpy pandas notebook ipywidgets plotly "transformers<=5.5.0" unsloth_zoo xformers --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match;
"""


model_id = "Qwen/Qwen3-VL-2B-Instruct"

try:
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=256,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    """
    This part for some reason throws following exception
    RuntimeError: Direct module loading failed for 
    unsloth_compiled_module_qwen3_vl: invalid syntax 
    (unsloth_compiled_module_qwen3_vl.py, line 612) 
    """
except Exception:
    """But I can safely ignore it by recalling the function here"""
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=256,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )


fv_model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,  # freeze the ViT
    finetune_language_layers=True,  # adapt the LLM
    finetune_attention_modules=True,  # LoRA on q, k, v, o
    finetune_mlp_modules=True,  # LoRA on gate, up, down
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

visual_model = fv_model.visual
lm_model = fv_model.language_model.layers
lm_head = fv_model.lm_head

fv_model.print_trainable_parameters()

vision_parameters = {"total": 0, "trainable": 0}
for name, param in visual_model.named_parameters():
    num_params = param.numel()
    vision_parameters["total"] += num_params
    if param.requires_grad:
        vision_parameters["trainable"] += num_params

print(
    f"Vision part: trainable: {vision_parameters['trainable']} \
        | total: {vision_parameters['total']} \
        | % trainable : {vision_parameters['trainable'] / vision_parameters['total'] * 100}"
)

lm_model_parameters = {"total": 0, "trainable": 0}
for name, param in lm_model.named_parameters():
    num_params = param.numel()
    lm_model_parameters["total"] += num_params
    if param.requires_grad:
        lm_model_parameters["trainable"] += num_params

print(
    f"Lang part: trainable: {lm_model_parameters['trainable']:.2e} \
        | total: {lm_model_parameters['total']:.2e} \
        | % trainable : {lm_model_parameters['trainable'] / lm_model_parameters['total'] * 100:.2e}"
)

lm_head_parameters = {"total": 0, "trainable": 0}
for name, param in lm_head.named_parameters():
    num_params = param.numel()
    lm_head_parameters["total"] += num_params
    if param.requires_grad:
        lm_head_parameters["trainable"] += num_params

print(
    f"lm_head part: trainable: {lm_head_parameters['trainable']} \
        | total: {lm_head_parameters['total']} \
        | % trainable : {lm_head_parameters['trainable'] / lm_head_parameters['total'] * 100}"
)
