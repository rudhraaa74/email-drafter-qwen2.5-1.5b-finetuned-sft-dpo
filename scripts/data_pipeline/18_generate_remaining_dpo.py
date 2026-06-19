import os
import json
import time
from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "gemma-4-31b-it" 
INPUT_FILE = "sft_data/train.jsonl"
OUTPUT_FILE = "dpo_data/dpo_combined.jsonl"
TARGET_TOTAL = 250
RATE_LIMIT_DELAY = 0.1  # Relies on Tenacity exponential backoff for rate limits instead

# Setup Gemini client
client = genai.Client(api_key=API_KEY)

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=20))
def generate_dpo_pair(prompt_text):
    system_prompt = """You are generating synthetic preference data for an AI alignment task.
You must return a JSON object with two fields: "chosen" and "rejected".
Both should be valid responses to the user's prompt.

- "chosen": This should be an normal , direct, human-like email. It should be short, get straight to the point, and avoid ALL AI-isms. No "I hope this finds you well", no "Best regards", no robotic politeness.
- "rejected": This should be the classic "AI slop". Make it overly polite, verbose, formal, and include standard AI email tropes.

Output ONLY a raw JSON object like:
{
  "chosen": "...",
  "rejected": "..."
}"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=system_prompt + "\n\nUser Prompt: " + prompt_text
    )
    
    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
        
    try:
        pair = json.loads(text)
        if "chosen" in pair and "rejected" in pair:
            return pair
        else:
            raise ValueError(f"Missing keys in JSON. Raw text: {text}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}. Raw text: {text}")

def main():
    print(f"Loading prompts from {INPUT_FILE}...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    prompts = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Extract user prompt from SFT format
                prompt = data["messages"][0]["content"]
                prompts.append(prompt)
                
    print(f"Loaded {len(prompts)} prompts.")

    processed_prompts = set()
    current_total = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "prompt" in data:
                        processed_prompts.add(data["prompt"])
                        current_total += 1
                        
    print(f"Found {current_total} already processed in {OUTPUT_FILE}.")
    
    pairs_needed = TARGET_TOTAL - current_total
    if pairs_needed <= 0:
        print("Target of 250 pairs already reached.")
        return
        
    prompts_to_process = [p for p in prompts if p not in processed_prompts][:pairs_needed]
    print(f"Remaining to generate: {len(prompts_to_process)}")

    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    # We will process sequentially or in very small batches to avoid rate limits
    with open(OUTPUT_FILE, "a") as f:
        for i, prompt in enumerate(prompts_to_process, 1):
            try:
                pair = generate_dpo_pair(prompt)
                
                output_data = {
                    "prompt": prompt,
                    "chosen": pair["chosen"],
                    "rejected": pair["rejected"]
                }
                f.write(json.dumps(output_data) + "\n")
                f.flush()
                
                success_count += 1
                if i % 10 == 0:
                    print(f"Processed {i}/{len(prompts_to_process)}... ({fail_count} failed)")
                
                time.sleep(RATE_LIMIT_DELAY)
                    
            except Exception as exc:
                fail_count += 1
                if "RetryError" in str(type(exc).__name__):
                    print(f"Prompt {i} generated RetryError. Inner exception: {exc.last_attempt.exception()}")
                else:
                    print(f"Prompt {i} generated an exception: {type(exc).__name__} - {exc}")

    end_time = time.time()
    print(f"\nFinished in {end_time - start_time:.2f} seconds.")
    print(f"Total Success: {success_count}, Total Failed: {fail_count}")

if __name__ == "__main__":
    main()
