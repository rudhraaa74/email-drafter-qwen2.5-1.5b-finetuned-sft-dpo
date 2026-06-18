import os
from google import genai
from google.genai import types

# 1. Hardcode the API key directly
api_key = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
client = genai.Client(api_key=api_key)

# 2. Setup the test email we just tried with Mistral
test_email = "Hi Lee, Pete Thompson is preparing the LV Co Gen break out contract based on the version generated last week that you and Bill Williams had agreed to (or so I think), but incorporating the AIP stuff where approproiate for the new turbines. We need to have the Delta break out contract signed by April 11th, so I'm hoping to have the LV Co Gen contract today or early tomorrow for all to review. That way, you can look at both at the same time, and sign both at the same time. Pretty nifty! Thanks,"

# 3. Setup the basic prompt
prompt = f"""Read this email and write the short casual message the sender would have typed into an AI to get it written. First person, conversational, include key details like names and dates. Output only the raw prompt text, no options, no explanations, no quotes around it

Email:
{test_email}

Prompt:
"""

print("\\nGenerating with Gemini...")
response = client.models.generate_content(
    model='gemma-4-31b-it',
    contents=prompt
)

# 4. Print the result
print("\\n" + "="*50)
print(f"INPUT EMAIL:\\n{test_email}")
print("-" * 50)
print(f"GENERATED PROMPT:\\n{response.text.strip()}")
print("="*50)
