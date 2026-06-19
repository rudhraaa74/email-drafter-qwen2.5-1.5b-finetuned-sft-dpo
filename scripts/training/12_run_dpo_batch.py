import json
import os
import sys
import mlx.core as mx
from datasets import Dataset
from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig, prepare_preference_dataset

def main():
    pairs_file = "dpo_data/dpo_pairs.jsonl"
    adapter_dir = "adapters/dpo_adapter"
    base_model = "models/Qwen2.5-1.5B-Instruct"

    if not os.path.exists(pairs_file):
        print("No pairs file found.")
        sys.exit(0)

    # 1. Load the most recent pairs (all pairs so far)
    pairs = []
    with open(pairs_file, "r") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
                
    if len(pairs) == 0:
        print("No pairs to train on.")
        sys.exit(0)

    # 2. Convert to Chat format
    hf_pairs = []
    for p in pairs:
        hf_pairs.append({
            "chosen": [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["chosen"]}],
            "rejected": [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": p["rejected"]}]
        })

    dataset = Dataset.from_list(hf_pairs)

    # 3. Load model and tokenizer
    print("Loading model and adapter into DPOTrainer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model,
        max_seq_length=1024,
        adapter_path=adapter_dir if os.path.exists(os.path.join(adapter_dir, "adapters.safetensors")) else None
    )

    # If it didn't have adapters, apply PEFT
    if not os.path.exists(os.path.join(adapter_dir, "adapters.safetensors")):
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=True
        )

    # 4. Format Dataset
    formatted_ds = prepare_preference_dataset(dataset, tokenizer, format_type="dpo")

    # 5. DPO Config for rapid burst training
    config = DPOConfig(
        beta=0.1,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=20, # 20 steps of rapid update
        save_steps=1000, # We'll save manually
        logging_steps=5,
        output_dir="adapters/dpo_temp",
        max_seq_length=1024,
        max_prompt_length=512
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=formatted_ds,
        tokenizer=tokenizer,
        args=config
    )

    print("\n--- Starting Batched DPO Update ---")
    trainer.train()

    print("\nSaving Updated Adapter...")
    os.makedirs(adapter_dir, exist_ok=True)
    # MLX-Tune trainer saves adapters via save_model_hf_format or similar, but we can just use mlx_lm to save.
    # Actually DPOTrainer has a save_model method? Let's check if the trainer saves automatically.
    # We will just let mlx-tune handle saving. Wait, mlx-tune trainer uses `model.save_pretrained`.
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_dir)
    else:
        # Fallback to MLX-LM save
        import mlx_lm
        import mlx.core as mx
        # Save weights
        weights = {}
        for k, v in model.parameters().items():
            if "lora" in k:
                weights[k] = v
        mx.save_safetensors(os.path.join(adapter_dir, "adapters.safetensors"), weights)
        # We don't change adapter_config.json since it's the same shape

    print("Batched Update Complete! Freeing VRAM...")

if __name__ == "__main__":
    main()
