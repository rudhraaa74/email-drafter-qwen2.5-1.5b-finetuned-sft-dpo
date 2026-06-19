import json
import time
import os
import argparse
from tqdm import tqdm
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Generate responses using base Qwen model.")
    parser.add_argument("--input", type=str, default="eval_data/eval.jsonl", help="Input JSONL file with prompts")
    parser.add_argument("--output", type=str, default="eval_data/base_eval_outputs.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts to process")
    args = parser.parse_args()

    print(f"Loading base model from models/Qwen2.5-1.5B-Instruct...")
    model, tokenizer = load("models/Qwen2.5-1.5B-Instruct")
    
    # Read prompts
    prompts = []
    with open(args.input, "r") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            # Support both eval.jsonl format and the raw dataset_with_prompts.jsonl format
            p = item.get("prompt") or item.get("generated_prompt")
            if p:
                prompts.append((item, p))
                
    if args.limit:
        prompts = prompts[:args.limit]
        
    print(f"Loaded {len(prompts)} prompts from {args.input}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    start_time = time.time()
    
    with open(args.output, "w") as f_out:
        for item, prompt_text in tqdm(prompts, desc="Generating"):
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            # Generate at default temperature
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=300, 
                verbose=False
            )
            
            output_item = item.copy()
            output_item["base_model_response"] = response.strip()
            
            f_out.write(json.dumps(output_item) + "\n")
            f_out.flush()

    elapsed = time.time() - start_time
    print(f"\nDone! Generated {len(prompts)} responses in {elapsed:.2f} seconds.")
    print(f"Average time per prompt: {elapsed/len(prompts):.2f}s")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
