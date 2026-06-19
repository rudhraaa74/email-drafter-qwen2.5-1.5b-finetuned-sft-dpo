import os
import sys
import json
from datasets import Dataset
from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig, prepare_preference_dataset

def main():
    pairs_file = "dpo_data/dpo_combined.jsonl"
    base_model = "models/Qwen2.5-1.5B-Instruct-SFT"
    adapter_dir = "adapters/dpo_adapter_final"

    print("Loading pairs from dataset...")
    pairs = []
    with open(pairs_file, "r") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    print(f"Loading base model {base_model}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model,
        max_seq_length=1024,
    )

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

    formatted_ds = prepare_preference_dataset(dataset, tokenizer, format_type="dpo")

    # Configuration for 1 epoch over 250 pairs (63 iterations with effective batch size 4)
    config = DPOConfig(
        beta=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=63,
        logging_steps=10,
        save_steps=63,
        output_dir="adapters/dpo_temp_run_final",
        max_seq_length=1024,
        max_prompt_length=512
    )

    trainer = DPOTrainer(
        model=model,
        train_dataset=formatted_ds,
        tokenizer=tokenizer,
        args=config
    )

    print("\n--- Starting DPO Training ---")
    trainer.train()

    print(f"\nSaving final DPO adapters to {adapter_dir}...")
    os.makedirs(adapter_dir, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_dir)
    else:
        import mlx.core as mx
        weights = {}
        for k, v in model.parameters().items():
            if "lora" in k:
                weights[k] = v
        mx.save_safetensors(os.path.join(adapter_dir, "adapters.safetensors"), weights)

    print("DPO Training Complete!")

if __name__ == "__main__":
    main()
