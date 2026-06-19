import sys
from mlx_lm import load, generate

def main():
    print("="*60)
    print("Loading base model Qwen/Qwen2.5-1.5B-Instruct...")
    print("This will take a few seconds...")
    print("="*60)
    
    # Load the base model and tokenizer from local folder
    model, tokenizer = load("models/Qwen2.5-1.5B-Instruct")
    
    print("\nModel loaded successfully!")
    print("This is the completely UN-FINETUNED base model.")
    print("Test it out to see how robotic it sounds and note down the 'slop' words.")
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
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            print("\nResponse:\n" + "-"*60)
            # Generate and print the response
            # verbose=True streams the output token by token!
            generate(model, tokenizer, prompt=prompt, max_tokens=300, verbose=True)
            print("-" * 60)
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
