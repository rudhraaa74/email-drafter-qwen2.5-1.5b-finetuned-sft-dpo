import json
import time
import os
import argparse
from tqdm import tqdm
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Generate responses using SFT Qwen model.")
    parser.add_argument("--model", type=str, default="models/Qwen2.5-1.5B-Instruct", help="Base model path")
    parser.add_argument("--adapter", type=str, default="adapters/sft_adapter", help="Adapter path")
    parser.add_argument("--input", type=str, default="sft_data/test.jsonl", help="Input JSONL file with prompts")
    parser.add_argument("--output", type=str, default="eval_outputs/sft_eval_outputs.jsonl", help="Output JSONL file")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of prompts to process")
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
            # The test.jsonl file contains full chat format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
            # We want to extract just the user prompt
            messages = item.get("messages", [])
            if not messages:
                # Fallback if the format is just simple prompts
                p = item.get("prompt") or item.get("generated_prompt")
                if p:
                    prompts.append((item, p))
            else:
                user_msg = [m for m in messages if m["role"] == "user"]
                if user_msg:
                    prompts.append((item, user_msg[0]["content"]))
                
    if args.limit:
        prompts = prompts[:args.limit]
        
    print(f"Loaded {len(prompts)} prompts from {args.input}")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    start_time = time.time()
    
    with open(args.output, "w") as f_out:
        for item, prompt_text in tqdm(prompts, desc="Generating"):
            messages = [{"role": "user", "content": prompt_text}]
            # We add format properly without the assistant response
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate at default temperature
            response = generate(
                model, 
                tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=300, 
                verbose=False
            )
            
            # Save format matching evaluation
            output_item = {"prompt": prompt_text, "sft_model_response": response.strip(), "base_model_response": response.strip()}
            # Note: We duplicate it into 'base_model_response' temporarily because evaluate.py expects it for scoring.
            # Actually, evaluate.py just expects the response to be in the values, we can configure evaluate.py to read either.
            
            f_out.write(json.dumps(output_item) + "\n")
            f_out.flush()

    elapsed = time.time() - start_time
    print(f"\nDone! Generated {len(prompts)} responses in {elapsed:.2f} seconds.")
    print(f"Average time per prompt: {elapsed/len(prompts):.2f}s")
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
