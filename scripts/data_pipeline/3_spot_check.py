import json
import random

def main():
    input_path = "raw_data/aeslc_cleaned.jsonl"
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    sample = random.sample(lines, min(10, len(lines)))
    
    print("\n--- SPOT CHECK (10 Sampled Emails) ---\n")
    for i, line in enumerate(sample):
        email = json.loads(line).get('email', '')
        print(f"=== Email {i+1} ===")
        print(email)
        print("====================\n")

if __name__ == '__main__':
    main()
