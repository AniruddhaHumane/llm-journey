from build_dataset import build_dataset, lazy_load_batch
from unsloth import FastVisionModel, UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen3-VL-2B-Instruct"

try:
    model, processor = FastVisionModel.from_pretrained(
        model_id,
        max_seq_length=2048,
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
        max_seq_length=2048,
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
FastVisionModel.for_training(fv_model)

collator = UnslothVisionDataCollator(
    model=fv_model,
    processor=processor,
    max_seq_length=2048,
    train_on_responses_only=True,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

train_dataset, val_dataset, test_dataset = build_dataset(verbose=False)
train_dataset = train_dataset.with_transform(lazy_load_batch)
val_dataset = val_dataset.with_transform(lazy_load_batch)

# args = SFTConfig(
#     output_dir="./results",
#     max_length=2048,  # based on token analysis
#     # --- training and validation ---
#     # num_train_epochs=1,
#     max_steps=1,
#     per_device_train_batch_size=1,
#     per_device_eval_batch_size=1,
#     gradient_accumulation_steps=1,
#     # --- Loss & Metrics Logging ---
#     logging_steps=1,  # Log training loss every 10 steps
#     eval_strategy="steps",  # Run evaluation periodically
#     eval_steps=1,  # Evaluate loss on validation set every 50 steps
#     save_strategy="steps",  # Save model checkpoints
#     save_steps=1,
#     # ---- dataset preprocessing
#     remove_unused_columns=False,
#     dataset_text_field="",
#     dataset_kwargs={"skip_prepare_dataset": True},  # <--- Fixes transform loss
# )

# config for training
args = SFTConfig(
    output_dir="./results",
    max_length=2048,  # Based on token analysis
    # --- Training Epochs & Batching ---
    num_train_epochs=3,  # Run for full passes over training dataset
    max_steps=-1,  # Disabled (-1 lets num_train_epochs control training duration)
    per_device_train_batch_size=4,  # Adjust based on GPU VRAM (4 or 2)
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
    # --- Learning Rate & Optimizer ---
    learning_rate=2e-4,  # Standard learning rate for LoRA/QLoRA
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,  # Set to fp16=True if Ampere GPU (A100/3090/4090) is not available
    # --- Loss & Metrics Logging ---
    logging_steps=10,  # Log metrics every 10 steps
    eval_strategy="epoch",  # Run evaluation at the end of every epoch
    save_strategy="epoch",  # Save checkpoint at the end of every epoch
    save_total_limit=2,  # Keep only the last 2 checkpoints to conserve disk space
    load_best_model_at_end=True,  # Load the best performing checkpoint after training
    metric_for_best_model="eval_loss",
    # --- Dataset Preprocessing ---
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},  # Keeps custom vision transforms
)

trainer = SFTTrainer(
    fv_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=args,
    data_collator=collator,
)

# confirming where loss will be calculated:
batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
input_ids = batch["input_ids"][0]
# Extract unmasked tokens where loss will be computed
active_tokens = labels[labels != -100]
loss_text = processor.tokenizer.decode(active_tokens, skip_special_tokens=False)
print("=== TOKENS COMPUTING LOSS ===")
print(loss_text)


trainer.train()
