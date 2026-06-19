import os
import sys
import mlx.core as mx
from datasets import Dataset
from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig, prepare_preference_dataset
import json

def main():
    pairs_file = sys.argv[1] if len(sys.argv) > 1 else "dpo_data/dpo_pairs.jsonl"
    adapter_dir = "adapters/dpo_adapter"
    base_model = "models/Qwen2.5-1.5B-Instruct"

    if not os.path.exists(pairs_file):
        print(f"No pairs file found at {pairs_file}.")
        sys.exit(1)

    print("Loading pairs from dataset...")
    pairs = []
    with open(pairs_file, "r") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
                
    if len(pairs) == 0:
        print("No pairs to train on.")
        sys.exit(1)

    print(f"\nLoading base model {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model,
        max_seq_length=1024,
    )

    print("\nApplying LoRA Adapters for DPO...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Convert to strings
    hf_pairs = []
    for p in pairs:
        prompt_str = tokenizer.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True)
        hf_pairs.append({
            "prompt": prompt_str,
            "chosen": p["chosen"] + "<|im_end|>\n",
            "rejected": p["rejected"] + "<|im_end|>\n"
        })

    dataset = Dataset.from_list(hf_pairs)
    print(f"Loaded {len(dataset)} training pairs.")

    print("Formatting preference dataset...")
    formatted_ds = prepare_preference_dataset(dataset, tokenizer, format_type="dpo")

    config = DPOConfig(
        beta=0.1,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=15,
        logging_steps=5,
        save_steps=500,
        output_dir="adapters/dpo_temp_offline",
        max_seq_length=1024,
        max_prompt_length=512
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=formatted_ds,
        tokenizer=tokenizer,
        args=config
    )

    print("\n--- Starting Full Offline DPO Training ---")
    trainer.train()

    print(f"\nSaving final DPO adapters to {adapter_dir}...")
    os.makedirs(adapter_dir, exist_ok=True)
    
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_dir)
    else:
        # Fallback to MLX-LM save
        import mlx_lm
        import mlx.core as mx
        weights = {}
        for k, v in model.parameters().items():
            if "lora" in k:
                weights[k] = v
        mx.save_safetensors(os.path.join(adapter_dir, "adapters.safetensors"), weights)

    print("DPO Training Complete!")

if __name__ == "__main__":
    main()
