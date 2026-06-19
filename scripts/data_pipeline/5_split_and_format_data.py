import json
import random
import os

INPUT_FILE = "processed_data/dataset_with_prompts.jsonl"
TRAIN_FILE = "sft_data/train.jsonl"
VALID_FILE = "sft_data/valid.jsonl"
EVAL_FILE = "eval_data/eval.jsonl"

os.makedirs("sft_data", exist_ok=True)
os.makedirs("eval_data", exist_ok=True)

# Load data
with open(INPUT_FILE, "r") as f:
    data = [json.loads(line.strip()) for line in f if line.strip()]

# Remove invalid items
valid_data = []
for item in data:
    if "generated_prompt" in item and "email" in item:
        valid_data.append(item)

# Shuffle with fixed seed
random.seed(42)
random.shuffle(valid_data)

# 1. Carve out exactly 100 for eval (only saving the prompt)
eval_data = valid_data[:100]
remaining_data = valid_data[100:]

# 2. Format remaining into chat format
formatted_remaining = []
for item in remaining_data:
    formatted_remaining.append({
        "messages": [
            {"role": "user", "content": item["generated_prompt"]},
            {"role": "assistant", "content": item["email"]}
        ]
    })

# 3. Split remaining into train/valid (90/10, valid capped at 200)
num_valid = min(int(len(formatted_remaining) * 0.1), 200)
valid_split = formatted_remaining[:num_valid]
train_split = formatted_remaining[num_valid:]

# Save Eval (only prompt)
with open(EVAL_FILE, "w") as f:
    for item in eval_data:
        json.dump({"prompt": item["generated_prompt"]}, f)
        f.write("\n")

# Save Train
with open(TRAIN_FILE, "w") as f:
    for item in train_split:
        json.dump(item, f)
        f.write("\n")

# Save Valid
with open(VALID_FILE, "w") as f:
    for item in valid_split:
        json.dump(item, f)
        f.write("\n")

print(f"Total rows: {len(valid_data)}")
print(f"Eval rows: {len(eval_data)}")
print(f"Train rows: {len(train_split)}")
print(f"Valid rows: {len(valid_split)}")
