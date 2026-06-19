import sys
from mlx_lm import load, generate

def main():
    print("="*60)
    print("Loading base model Qwen2.5-1.5B-Instruct-SFT + New DPO Adapter...")
    print("This will take a few seconds...")
    print("="*60)
    
    # Load the base model and tokenizer with the DPO adapter
    model, tokenizer = load("models/Qwen2.5-1.5B-Instruct-SFT", adapter_path="adapters/dpo_temp_run_final/adapters")
    
    print("\nModel loaded successfully!")
    print("This is the newly trained DPO model.")
    print("Type your prompt below. Type 'quit' or 'exit' to stop.")
    
    while True:
        try:
            user_input = input("\nPrompt> ")
            if user_input.strip().lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue
            
            # Apply chat template
            messages = [{"role": "user", "content": user_input}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            print("\nResponse:\n" + "-"*60)
            # Generate and print the response
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(0.5)
            generate(model, tokenizer, prompt=prompt, max_tokens=300, sampler=sampler, verbose=True)
            print("\n" + "-" * 60)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
