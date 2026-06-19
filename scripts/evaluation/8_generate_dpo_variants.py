import json
import time
import os
import argparse
from tqdm import tqdm
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Generate 5 variants per prompt for DPO ranking.")
    parser.add_argument("--model", type=str, default="models/Qwen2.5-1.5B-Instruct", help="Base model path")
    parser.add_argument("--adapter", type=str, default="adapters/sft_adapter", help="Adapter path")
    parser.add_argument("--input", type=str, default="eval_data/eval.jsonl", help="Input JSONL file with prompts")
    parser.add_argument("--output", type=str, default="dpo_data/variants.jsonl", help="Output JSONL file")
    parser.add_argument("--num_variants", type=int, default=5, help="Number of variants per prompt")
    parser.add_argument("--temp", type=float, default=0.85, help="Temperature for generation")
    args = parser.parse_args()

    print(f"Loading base model from {args.model}")
    print(f"Loading adapter from {args.adapter}")
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    
    # Read prompts
    prompts = []
    with open(args.input, "r") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Extract prompt depending on JSON format
            p = item.get("prompt") or item.get("generated_prompt")
            if not p and "messages" in item:
                user_msg = [m for m in item["messages"] if m["role"] == "user"]
                if user_msg: p = user_msg[0]["content"]
                    
            if p:
                prompts.append(p)
                
    print(f"Loaded {len(prompts)} prompts from {args.input}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    start_time = time.time()
    
    with open(args.output, "w") as f_out:
        for prompt_text in tqdm(prompts, desc="Generating variants"):
            messages = [{"role": "user", "content": prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            variants = []
            for _ in range(args.num_variants):
                # Generate at higher temperature for stylistic variance
                response = generate(
                    model, 
                    tokenizer, 
                    prompt=formatted_prompt, 
                    max_tokens=300, 
                    temp=args.temp,
                    verbose=False
                )
                variants.append(response.strip())
            
            output_item = {
                "prompt": prompt_text,
                "variants": variants
            }
            
            f_out.write(json.dumps(output_item) + "\n")
            f_out.flush()

    elapsed = time.time() - start_time
    print(f"\nDone! Generated {len(prompts)}x{args.num_variants} variants in {elapsed:.2f} seconds.")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
