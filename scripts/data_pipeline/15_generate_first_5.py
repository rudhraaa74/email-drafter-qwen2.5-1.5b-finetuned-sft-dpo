import os
import json
from google import genai

API_KEY = "YOUR_API_KEY_HERE"
MODEL_NAME = "gemma-4-31b-it"

def main():
    print("Loading first 5 prompts from train.jsonl...")
    prompts = []
    with open("sft_data/train.jsonl", "r") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            if line.strip():
                data = json.loads(line)
                prompts.append(data["messages"][0]["content"])
                
    client = genai.Client(api_key=API_KEY)
    
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

    print("\n--- Starting Generation ---\n")
    for i, prompt in enumerate(prompts, 1):
        print(f"Prompt {i}:\n{prompt}")
        print("Waiting for Gemma response...")
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=system_prompt + "\n\nUser Prompt: " + prompt
            )
            print("Gemma Response:")
            print(response.text.strip())
        except Exception as e:
            print(f"Error calling API: {type(e).__name__} - {e}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()
