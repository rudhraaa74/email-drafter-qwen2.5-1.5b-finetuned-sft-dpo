import os
import json
import time
import requests
import argparse

# Default prompt for the LLM
SYSTEM_PROMPT = """You are an expert dataset generator for AI alignment.
Your task is to generate pairs of emails to train an AI on how to be extremely concise and direct.

Generate 10 random modern workplace scenarios (e.g., Slack messages, Zoom invites, Jira ticket updates, AWS downtime, vacation requests, marketing briefs).
For each scenario, provide:
1. "prompt": The user's instructions (e.g., "Draft an email to Mark asking for the TPS reports").
2. "chosen": The ideal response. It MUST be extremely concise, direct, under 3 sentences, and strip out all unnecessary pleasantries like "I hope this finds you well" or "Best regards".
3. "rejected": The "slop" response. It should be verbose, robotic, full of AI padding, overly polite, and annoyingly long.

Output strictly as a JSON list of objects. Each object must have "prompt", "chosen", and "rejected" string keys.
Do not output markdown blocks or any other text, just the raw JSON array.
"""

def generate_batch(api_key, model_name="gemma-2-27b-it"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    
    # Google AI Studio format
    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": "Generate 10 pairs. Output only raw JSON."}]}
        ],
        "generationConfig": {
            "temperature": 0.9,
            "responseMimeType": "application/json"
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"API Error: {response.text}")
        return []
        
    try:
        content = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # Sometimes models wrap in markdown despite instructions
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        data = json.loads(content)
        
        # Handle cases where the LLM wraps the array in a dict (e.g., {"pairs": [...]})
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return []
        return data
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=os.environ.get("GEMMA_API_KEY", ""), help="API Key for the provider")
    parser.add_argument("--model", type=str, default="gemma-2-27b-it", help="Model name to use")
    parser.add_argument("--target-pairs", type=int, default=250, help="Number of pairs to generate")
    parser.add_argument("--output", type=str, default="dpo_data/dpo_google_gemma.jsonl", help="Output file")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Please provide an API key via --api-key or the GEMMA_API_KEY environment variable.")
        sys.exit(1)

    print(f"Starting generation to target {args.target_pairs} pairs...")
    print(f"Using model: {args.model} via Google AI Studio")
    print(f"Output file: {args.output}")
    
    collected_pairs = 0
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, "a") as f:
        while collected_pairs < args.target_pairs:
            print(f"[{collected_pairs}/{args.target_pairs}] Requesting batch...")
            
            pairs = generate_batch(args.api_key, args.model)
            if not pairs:
                print("Empty or invalid batch. Retrying in 2 seconds...")
                time.sleep(2)
                continue
                
            for pair in pairs:
                if all(k in pair for k in ["prompt", "chosen", "rejected"]):
                    f.write(json.dumps(pair) + "\n")
                    collected_pairs += 1
                    
            print(f"Successfully added {len(pairs)} pairs. Total: {collected_pairs}")
            time.sleep(1) # Simple rate limiting

    print(f"Done! Generated {collected_pairs} pairs and saved to {args.output}")

if __name__ == "__main__":
    import sys
    main()
