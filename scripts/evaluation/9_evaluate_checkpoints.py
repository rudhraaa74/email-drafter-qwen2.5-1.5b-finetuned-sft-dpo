import os
import glob
import re
from mlx_lm import load, generate

PROMPTS = [
    "Write a short email asking for a project update.",
    "Write an email requesting a 3-day deadline extension from your manager.",
    "Write a short follow-up email asking whether someone has reviewed a proposal.",
    "Write an email informing a colleague that a meeting needs to be rescheduled.",
    "Write a short email asking a teammate for feedback on a document."
]

def main():
    adapter_dir = "adapters/sft_adapter"
    evals_dir = "evals"
    os.makedirs(evals_dir, exist_ok=True)
    
    # Find all checkpoint files
    # Checkpoints format: 0000200_adapters.safetensors, etc. 
    # Also "adapters.safetensors" for the final one, which is step 2000.
    
    checkpoints = {}
    
    for fname in os.listdir(adapter_dir):
        if not fname.endswith(".safetensors"): continue
        if fname == "adapters.safetensors":
            checkpoints[2000] = os.path.join(adapter_dir, fname)
        elif fname.endswith("_adapters.safetensors"):
            try:
                step_str = fname.replace("_adapters.safetensors", "")
                step = int(step_str)
                checkpoints[step] = os.path.join(adapter_dir, fname)
            except ValueError:
                pass
                
    if not checkpoints:
        print("No checkpoints found!")
        return
        
    print(f"Found checkpoints for steps: {sorted(checkpoints.keys())}")
    
    base_model_path = "models/Qwen2.5-1.5B-Instruct"
    
    # Ensure memory is clean before looping
    
    for step in sorted(checkpoints.keys()):
        adapter_path = checkpoints[step]
        # In mlx_lm, we pass the directory as the adapter_path, but wait... 
        # Actually, `mlx_lm` expects the directory, and inside the directory it expects `adapters.safetensors`
        # Since the checkpoints are stored as `0000200_adapters.safetensors` in the same directory,
        # we need to rename or copy them so `mlx_lm` can load them properly?
        # Actually, `mlx_lm` has `adapter_file` parameter in `load` but the public API might just take `adapter_path`.
        # Let's check `mlx_lm.load` signature. 
        # MLX-LM `load` natively supports `adapter_path` pointing to a file or directory. 
        # If it's a file, it loads that file. Let's just point to the .safetensors file directly.
        # Wait, mlx_lm checks if the path is a dir. If it's a file, it might fail. 
        # The safest way is to pass the specific file. Wait, in `mlx_lm.load`, if `adapter_path` is provided,
        # it usually looks for `adapter_config.json` inside its dirname.
        
        out_file = os.path.join(evals_dir, f"checkpoint_{step}.txt")
        print(f"Evaluating checkpoint {step} -> {out_file}")
        
        # Load the model and specific adapter file
        try:
            # We must load with the adapter file. But mlx_lm `load` only takes `adapter_path` which expects a directory.
            # To load a specific checkpoint, we might need to overwrite `adapters.safetensors` temporarily or use the CLI.
            # Let's temporarily copy the checkpoint to a temp directory to load it cleanly.
            
            import shutil
            temp_dir = f"adapters/temp_eval_{step}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy config and weights
            config_src = os.path.join(adapter_dir, "adapter_config.json")
            if os.path.exists(config_src):
                shutil.copy(config_src, os.path.join(temp_dir, "adapter_config.json"))
            shutil.copy(adapter_path, os.path.join(temp_dir, "adapters.safetensors"))
            
            model, tokenizer = load(base_model_path, adapter_path=temp_dir)
            
            with open(out_file, "w") as f_out:
                f_out.write("="*50 + "\n")
                f_out.write(f"CHECKPOINT: {step}\n")
                f_out.write("="*50 + "\n\n")
                
                for prompt_text in PROMPTS:
                    f_out.write("PROMPT:\n")
                    f_out.write(prompt_text + "\n\n")
                    
                    f_out.write("RESPONSE:\n")
                    messages = [{"role": "user", "content": prompt_text}]
                    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    response = generate(
                        model, 
                        tokenizer, 
                        prompt=formatted_prompt, 
                        max_tokens=300, 
                        verbose=False
                    )
                    
                    f_out.write(response.strip() + "\n\n")
                    f_out.write("="*50 + "\n\n")
                    
            # Cleanup temp dir
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"Failed to evaluate checkpoint {step}: {e}")

if __name__ == "__main__":
    main()
