import os
os.system("pip install -q -U accelerate")

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def main():
    model_name = "/kaggle/input/models/mistral-ai/mistral/pytorch/7b-v0.1-hf/1"
    print("Loading Mistral 7B Base in native 16-bit...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    input_path = "/kaggle/input/datasets/rudhrakoul/email-drafting/aeslc_cleaned.jsonl"
    output_path = "/kaggle/working/sft_data.jsonl"
    
    emails = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            emails.append(json.loads(line)['email'])
            
    # ONLY KEEP 15 PROMPTS FOR TESTING
    emails = emails[:15]
            
    print("Formatting prompts...")
    formatted_prompts = []
    for email in emails:
        prompt_text = f"""You will be provided with an email. Your task is to write the prompt a normal, hurried human would have typed into an AI to generate this email. The prompt should be realistic, conversational, and imperfect—exactly how a real user types. Reply ONLY with the prompt a user would type and nothing else.

Email:
Can you send me the November scheduling report when you get a chance? I need it before the team meeting on Thursday.

Prompt:
Write an email asking to send me the November report as I need it for my team meet on Thursday, be polite.

Email:
I wanted to let you know that the Henderson contract has been signed and returned. We should be good to proceed with the project kickoff next week.

Prompt:
Make an email to inform that Henderson's contract is signed and returned and we should be fine for the project kickoff next week, be casual.

Email:
Just a reminder that the building will be closed on Friday due to the holiday. Please make sure to submit your timesheets by Thursday noon.

Prompt:
Draft an email reminding that the building is closed because of holiday on Friday, submit timesheets before Thursday noon.

Email:
{email}

Prompt:
"""
        formatted_prompts.append(prompt_text)
        
    results = []
    batch_size = 8
    
    print(f"Starting batched generation with batch_size={batch_size}...")
    
    for i in tqdm(range(0, len(formatted_prompts), batch_size)):
        batch_prompts = formatted_prompts[i:i+batch_size]
        batch_emails = emails[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=True, 
                temperature=0.3,
                pad_token_id=tokenizer.pad_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        raw_responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for raw_response, email in zip(raw_responses, batch_emails):
            # Base models try to generate the next few-shot example. Split on "Email:" to stop it.
            response = raw_response.split("Email:")[0].strip()
            
            results.append({
                "prompt": response,
                "email": email
            })
            
        # Checkpoint every 10 batches
        if (i // batch_size) % 10 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                for row in results:
                    f.write(json.dumps(row) + "\n")
                    
    # Final save
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row) + "\n")
            
    print("Generation complete!")

if __name__ == "__main__":
    main()
