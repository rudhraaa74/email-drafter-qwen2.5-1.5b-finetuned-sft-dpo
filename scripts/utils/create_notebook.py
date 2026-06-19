import nbformat as nbf

nb = nbf.v4.new_notebook()

cell1 = nbf.v4.new_markdown_cell("# 1. Setup Environment\nInstall latest accelerate.")
cell2 = nbf.v4.new_code_cell("!pip install -q -U accelerate")

cell3 = nbf.v4.new_markdown_cell("# 2. Load Model and Tokenizer\nAdd **Mistral 7B v0.1 HF** to your notebook via the Kaggle Models tab on the right. Then copy its folder path and paste it below.")
cell4 = nbf.v4.new_code_cell("""import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace this with the path you copied from the Kaggle right-hand sidebar
# Example: "/kaggle/input/mistral/pytorch/7b-v0.1-hf/1"
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
print("Model loaded successfully!")""")

cell5 = nbf.v4.new_markdown_cell("# 3. Setup Single Test Email\nSince base models don't support chat templates, we format everything as a single raw text document.")
cell6 = nbf.v4.new_code_cell("""test_email = "Hey team, just a heads up that the servers will be down for maintenance this weekend. Please make sure to push all your code by Friday 5PM so nothing gets lost."

# Raw text formatting for Base model few-shot completion
prompt_text = f\"\"\"You will be provided with an email. Your task is to write the prompt a normal, hurried human would have typed into an AI to generate this email. The prompt should be realistic, conversational, and imperfect—exactly how a real user types. Reply ONLY with the prompt a user would type and nothing else.

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
{test_email}

Prompt:
\"\"\"

inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
print("Prompt formatted and ready for generation.")""")

cell7 = nbf.v4.new_markdown_cell("# 4. Generate and View Output\nRun the model on the single prompt and print the result.")
cell8 = nbf.v4.new_code_cell("""print("\\nGenerating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=64, 
        do_sample=True, 
        temperature=0.3,
        pad_token_id=tokenizer.pad_token_id
    )

input_length = inputs.input_ids.shape[1]
raw_response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

# Base models will try to continue the pattern and generate the NEXT email example. 
# We split on "Email:" to cut off anything the model hallucinates after the prompt.
response = raw_response.split("Email:")[0].strip()

print("\\n" + "="*50)
print(f"INPUT EMAIL:\\n{test_email}")
print("-" * 50)
print(f"GENERATED PROMPT:\\n{response}")
print("="*50)""")

nb['cells'] = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8]

with open('/Users/rudhrakoul/anti-ai-email-alignment/kaggle_notebook/generate_prompts.ipynb', 'w') as f:
    nbf.write(nb, f)
