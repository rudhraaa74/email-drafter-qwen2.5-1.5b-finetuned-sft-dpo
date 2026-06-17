import os
os.system("pip install -q -U bitsandbytes accelerate")

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

def main():
    model_name = "Qwen/Qwen2.5-14B-Instruct"
    print("Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    system_prompt = (
        "You are an expert executive assistant. You will be provided with an email. "
        "Your task is to write a single dense, imperative sentence that serves as a detailed instruction to write exactly this email. "
        "The instruction must include key specifics: the recipient or their role, the concrete ask, and any urgency, context, or tone derived from the email. "
        "Reply ONLY with the instruction sentence and nothing else. Do not use filler text like 'Here is the instruction:'."
    )
    
    few_shot_examples = [
        {
            "role": "user",
            "content": "Email:\nHi Sarah, please review the attached Q3 marketing budget and get back to me with your approval by EOD Thursday. It's urgent we finalize this before the Friday board meeting.\n\nInstruction:"
        },
        {
            "role": "assistant",
            "content": "Ask Sarah to urgently review and approve the attached Q3 marketing budget by EOD Thursday ahead of the Friday board meeting."
        },
        {
            "role": "user",
            "content": "Email:\nTeam, just a heads up that the office will be closing at 3 PM tomorrow due to the incoming snowstorm. Please work from home on Friday. Stay safe!\n\nInstruction:"
        },
        {
            "role": "assistant",
            "content": "Notify the team that the office closes early at 3 PM tomorrow due to the snowstorm and instruct them to work from home on Friday."
        },
        {
            "role": "user",
            "content": "Email:\nJohn, I am extremely disappointed with the delivery delay on the Miller project. We discussed this timeline extensively last month. Please call me immediately so we can figure out a recovery plan.\n\nInstruction:"
        },
        {
            "role": "assistant",
            "content": "Express severe disappointment to John regarding the Miller project delivery delay and demand an immediate phone call to establish a recovery plan."
        }
    ]
    
    input_path = "/kaggle/input/aeslc-cleaned-reverse-prompting/aeslc_cleaned.jsonl"
    output_path = "/kaggle/working/sft_data.jsonl"
    
    emails = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            emails.append(json.loads(line)['email'])
            
    print(f"Loaded {len(emails)} emails. Formatting prompts...")
    
    formatted_prompts = []
    for email in emails:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(few_shot_examples)
        messages.append({"role": "user", "content": f"Email:\n{email}\n\nInstruction:"})
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)
        
    results = []
    batch_size = 16
    
    print(f"Starting batched generation with batch_size={batch_size}...")
    
    for i in tqdm(range(0, len(formatted_prompts), batch_size)):
        batch_prompts = formatted_prompts[i:i+batch_size]
        batch_emails = emails[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id
            )
            
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        for response, email in zip(responses, batch_emails):
            results.append({
                "prompt": response.strip(),
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
