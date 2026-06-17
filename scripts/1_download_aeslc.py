import json
from datasets import load_dataset

def main():
    print("Loading Yale-LILY/aeslc dataset from HuggingFace...")
    ds = load_dataset("Yale-LILY/aeslc", split="train")
    
    output_path = "raw_data/aeslc_raw.jsonl"
    print(f"Saving {len(ds)} rows to {output_path}...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
            
    print("\n--- Schema Inspection (First 3 Rows) ---")
    for i in range(3):
        print(f"\nRow {i+1}:")
        print(json.dumps(ds[i], indent=2))

if __name__ == "__main__":
    main()
