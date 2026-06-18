import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.errors import APIError
from tenacity import retry, stop_after_attempt, wait_exponential

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_NAME = "gemma-4-26b-a4b-it"
INPUT_FILE = "kaggle_dataset/aeslc_cleaned.jsonl"
OUTPUT_FILE = "dataset_with_prompts.jsonl"
RATE_LIMIT_DELAY = 3.0 # Reduced sleep time to see if RPM increases

# Setup Gemini client
client = genai.Client(api_key=API_KEY)

# We use tenacity for backoffs. It will retry on all exceptions now, including DNS errors.
@retry(
    stop=stop_after_attempt(10), 
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def generate_prompt_for_email_batch(email_batch):
    emails_text = "\n\n---\n\n".join([f"Email {i+1}:\n{text}" for i, text in enumerate(email_batch)])
    prompt = f"""Read the following {len(email_batch)} emails. For each email, write the short casual message the sender would have typed into an AI to get it written. First person, conversational, include key details like names and dates.
Output your response ONLY as a valid JSON array of strings, with {len(email_batch)} strings.
Do not include any markdown formatting, explanations, or quotes outside the JSON array.

Emails:
{emails_text}

JSON Output:
"""
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
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
        prompts = json.loads(text)
        if isinstance(prompts, list) and len(prompts) == len(email_batch):
            return prompts
        else:
            raise ValueError(f"Expected a list of {len(email_batch)} strings, got {type(prompts)}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}. Raw text: {text}")

def main():
    print("Loading emails...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    emails = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.strip():
                emails.append(json.loads(line))
                
    print(f"Loaded {len(emails)} emails.")

    # Load existing prompts to resume if interrupted
    processed_emails = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "email" in data:
                        processed_emails.add(data["email"])
                        
    print(f"Found {len(processed_emails)} already processed.")
    
    emails_to_process = [e for e in emails if e["email"] not in processed_emails]
        
    batch_size = 2
    batches = [emails_to_process[i:i + batch_size] for i in range(0, len(emails_to_process), batch_size)]
    
    print(f"Processing {len(batches)} batches of up to {batch_size} emails...")

    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    # We open file in append mode to save incrementally
    with open(OUTPUT_FILE, "a") as f:
        for i, batch in enumerate(batches, 1):
            try:
                payload = [item["email"] for item in batch]
                prompts = generate_prompt_for_email_batch(payload)
                
                for item, prompt_text in zip(batch, prompts):
                    output_data = {
                        "email": item["email"],
                        "generated_prompt": prompt_text
                    }
                    f.write(json.dumps(output_data) + "\n")
                f.flush() # Ensure it writes immediately
                
                success_count += len(batch)
                print(f"Processed batch {i}/{len(batches)} (Total {success_count})... ({fail_count} failed)")
                
                # Sleep to respect rate limits
                time.sleep(RATE_LIMIT_DELAY)
                    
            except Exception as exc:
                fail_count += len(batch)
                print(f"Batch {i} generated an exception: {exc}")

    end_time = time.time()
    print(f"\nFinished in {end_time - start_time:.2f} seconds.")
    print(f"Total Success: {success_count}, Total Failed: {fail_count}")

if __name__ == "__main__":
    main()
