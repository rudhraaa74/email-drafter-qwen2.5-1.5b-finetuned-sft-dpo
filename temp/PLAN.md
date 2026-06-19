# Anti-AI Stylistic Alignment Pipeline
### Apple Silicon M1 — Edge Native via `mlx-lm`

---

## How to Use This Document

Each subphase is a single unit of work with one clear deliverable. Complete it, run the verify step, confirm it passes, then move to the next. Never skip a verify step. Hand this document to Antigravity one subphase at a time and ask it to write the code or command for that step only.

**Stack:** Python 3.11+, `mlx-lm`, HuggingFace `datasets`, Anthropic API (for prompt generation), Ollama  
**Hardware:** Apple Silicon M1 (8GB or 16GB unified memory)  
**Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`

---

## Phase 0 — Environment Bootstrap [DONE]

### 0.1 — Create a clean virtual environment
Create a new Python virtual environment dedicated to this project and activate it.

**Verify:** The `which python` command should return a path inside the new virtual environment, not the system Python.

---

### 0.2 — Install core dependencies
Install `mlx-lm`, `datasets`, `huggingface_hub`, and `tqdm` into the virtual environment using pip.

**Verify:** Import `mlx` and `datasets` in a one-liner Python check. Both should import without errors.

---

### 0.3 — Confirm DPO flag exists in mlx-lm-lora
Install `mlx-lm-lora` (which extends mlx-lm with DPO support). Run the `mlx_lm_lora.train --help` command and check that the `--train-mode` flag is listed with `dpo` as a valid option.

**Verify:** The help output includes `--train-mode` with `dpo` listed.


---

### 0.4 — Create project directory structure
Create a single top-level project folder with subfolders for scripts, raw data, SFT data, DPO data, eval data, adapter weights, evaluation outputs, and logs.

**Verify:** List the top-level folder and confirm all subfolders are present.

---

### 0.5 — Download base model weights
Write a short script that loads `Qwen/Qwen2.5-1.5B-Instruct` using `mlx_lm.load()`. This will cache the weights locally via HuggingFace.

**Verify:** The script runs without errors and prints a confirmation message. Weights are approximately 3GB and may take a few minutes on first download.

---

## Phase 1 — Data Curation [PARTIALLY DONE]

### 1.1 — Download the AESLC dataset [DONE]
Write a script that downloads the `aeslc` training split from HuggingFace using the `datasets` library and saves every row as a JSONL file in the raw data folder.

**Verify:** Count the lines in the output file. Expect approximately 18,000 rows.

---

### 1.2 — Inspect the raw data schema [DONE]
Print the first three rows of the raw JSONL file in a readable format to understand the field names and structure.

**Verify:** Identify the exact field names for the email body and subject line. Note them — they will be referenced in the next step.

---

### 1.3 — Strip metadata and filter by word count [DONE]
Write a script that reads the raw JSONL, removes forwarded and replied-to headers, removes metadata lines (To, From, Date, Subject, CC, and X- headers), truncates emails at the first line that contains legal boilerplate or confidentiality notices, and discards any email with fewer than 15 or more than 300 words. Finally, take a random sample capped at 2,000 rows (using a fixed random seed for reproducibility) and save the cleaned emails to a new JSONL file with a single `email` field per row.

**Verify:** Count the output file lines. Expect exactly 2,000 rows (or fewer if the dataset had fewer valid rows).

---

### 1.4 — Spot-check cleaned emails [DONE]
Write a script that randomly samples 10 emails from the cleaned file and prints each one.

**Verify (manual):** Read all 10. None should contain metadata headers, legal disclaimers, or obviously truncated sentences. If more than 2 out of 10 have issues, revisit the cleaning logic in 1.3.

---

### 1.5 — Generate reverse prompts via Gemini API [DONE]
Write a Python script (`generate_dataset_gemini.py`) that hits the Gemini API directly using the `gemma-4-26b-a4b-it` model. 

Key prompt generation constraints:
1. **Few-Shot Examples**: Use concrete few-shot examples in the system prompt rather than just negative constraints. This pattern-matches the desired format more reliably.
2. **Dense Instruction Phrasing**: The generated single-sentence imperative prompt must capture specific details from the email (recipient role, concrete ask, urgency, tone) rather than generic instructions like "Write an email to X".
3. **Robust Batched Generation**: Implement batched generation (processing 2 emails per prompt) and `tenacity` exponential backoff to handle 503/429 network and API errors resiliently, respecting the 15 RPM free tier limit.

Run the script locally to process all 2,000 emails and save the result as a new JSONL file (`dataset_with_prompts.jsonl`) with both a `prompt` field and an `email` field per row.

**Verify:** Count the output lines — should match the 2,000 file count.

---

### 1.6 — Spot-check generated prompts [DONE]
Write a script that randomly samples 5 rows from the prompts file and prints the prompt alongside the first 200 characters of the email.

**Verify (manual):** Prompts should be specific and task-oriented (e.g. "Request approval for the revised budget before end of quarter") not vague (e.g. "Write an email"). If more than 1 in 5 are vague or generic, revise the system prompt in step 1.5 and regenerate.

---

### 1.7 — Convert to SFT chat format and split into train and validation sets
Write a script that reads the prompts file, converts each row into a chat-style format with a `user` turn containing the prompt and an `assistant` turn containing the email, shuffles the rows, and splits them into a training set (90%) and a validation set (10%, capped at 200 rows). Save both as JSONL files in the SFT data folder.

**Verify:** Both files exist. Print and read one row from each — confirm the `messages` array contains the correct user and assistant turns.

---

### 1.8 — Build and lock the static evaluation set
Write a script that randomly samples exactly 100 rows from the prompts file using a fixed random seed and saves only the prompt text (no email body) to a separate eval JSONL file. This file must never be touched again after this step.

**Verify:** The eval file has exactly 100 lines. Confirm that none of these 100 prompts appear in the SFT training file.

---

## Phase 2 — Base Model Evaluation [DONE]

### 2.1 — Define the slop word list and metric functions
Write a Python module (not a script — a reusable module) that defines two functions. The first, `slop_count`, takes a text string and counts how many banned phrases it contains. The banned list should include words like: delve, robust, leverage, synergy, utilize, cutting-edge, game-changer, innovative, paradigm, transformative, holistic, actionable, seamlessly, empower, foster, stakeholder, deliverable, bandwidth, circle back, touch base, moving forward, going forward, at the end of the day, it is worth noting, it is important to note, in conclusion, in summary, i hope this email finds you well, please do not hesitate. The second function, `burstiness`, takes a text string, splits it into sentences, computes the length of each sentence in words, and returns the standard deviation of those lengths as a measure of pacing variance.

**Verify:** Run the module directly with a sample sentence that contains at least two slop phrases. Confirm `slop_count` returns 2 or more and `burstiness` returns a float.

---

### 2.2 — Generate base model outputs for all 100 eval prompts
Write a script that loads the un-finetuned base model (`Qwen2.5-1.5B-Instruct`), reads all 100 prompts from the eval file, generates one response per prompt at temperature 0.7 with a 300-token limit, and saves each prompt and its output to a JSONL file in the evals folder (e.g., `base_eval_outputs.jsonl`).

**Verify:** The output file has exactly 100 lines.

---

### 2.3 — Score base model outputs and save aggregate metrics
Write a script that reads the base evaluation outputs file, runs both metric functions from 2.1 on every output, computes the mean slop count, mean burstiness, and mean word count across all 100 outputs, and saves a summary JSON file. 

**Verify:** The summary JSON file exists and is readable. Write down the `mean_slop_count` and `mean_burstiness` values — this is your absolute "Base Baseline".

---

## Phase 3 — SFT Training & Evaluation [DONE]

### 3.1 — Dry-run SFT for 1 iteration
Run the `mlx_lm.lora` training command in SFT mode for exactly 1 iteration with a batch size of 2, pointing at the SFT data folder, and saving the adapter to a temporary dry-run path.

**Verify:** The command completes without errors and prints a loss value. If you get an out-of-memory error, reduce batch size to 1.

---

### 3.2 — Full SFT training run
Run the full SFT training command for 600 iterations with a batch size of 4, a learning rate of 1e-4, reporting every 50 steps, evaluating on the validation set every 100 steps, saving the adapter to the SFT adapters folder, and piping all output to a log file.

**Verify:** The adapter folder contains `adapter_config.json` and `adapters.safetensors`. Open the log file and confirm the validation loss at the end is lower than at the start. Expect the run to take 30–60 minutes on M1.

---

### 3.3 — Evaluate SFT Model
Reuse the evaluation scripts from Phase 2. First, generate outputs for the 100 eval prompts using the base model *with the new SFT adapter loaded*. Save to `sft_eval_outputs.jsonl`. Then, score the outputs using the metric script.

**Verify:** A new SFT summary JSON file is produced. Compare it to the base baseline to see how Supervised Fine-Tuning altered the slop count and burstiness.

---

## Phase 4 — Active DPO Data Collection [DONE]

### 4.1 — Generate 5 variants per eval prompt at high temperature
Write a script that loads the base model with the SFT adapter and, for each of the 100 eval prompts, generates 5 different responses at temperature 0.85 to encourage stylistic variance. Save all variants grouped by prompt to a JSONL file in the DPO data folder.

**Verify:** The variants file has exactly 100 lines. Each line contains a prompt and 5 response strings.

---

### 4.2 — Build an interactive CLI ranker with resumable progress
Write a script that reads the variants file and, for each prompt, displays all 5 variants numbered 1–5 and asks the user to rank them from best to worst by entering the numbers in order. The script should validate the input, save progress to a separate file after each prompt so the session can be interrupted and resumed, and build the final DPO pairs file by saving only the top-ranked variant as `chosen` paired against the 2nd- and 3rd-ranked variants as `rejected` (one pair each). The 4th and 5th ranked variants should be discarded as too ambiguous. The output should be a JSONL file where each line has `prompt`, `chosen`, and `rejected` fields.

**Verify:** After ranking 10 prompts, the DPO pairs file exists and contains approximately 20 rows. Interrupt and restart the script — confirm it resumes from prompt 11, not from the beginning.

**Target:** Rank all 100 prompts to produce approximately 200 high-quality pairs.

---

### 4.3 — Convert DPO pairs to training and validation splits
Write a script that reads the completed DPO pairs file, shuffles the rows, splits them into a training set (90%) and a validation set (the larger of 20 rows or 10%), and saves both to the DPO data folder.

**Verify:** Both files exist. Print one row from each and confirm the `prompt`, `chosen`, and `rejected` fields are all populated.

---

## Phase 5 — DPO Training [DONE]

### 5.1 — Dry-run DPO for 1 iteration
Run the `mlx_lm.lora` training command in DPO mode for exactly 1 iteration with a batch size of 1, pointing at the DPO data folder.

**Verify:** Completes without error and prints a loss value. If OOM, close background applications — batch size is already at minimum for DPO.

---

### 5.2 — Full DPO training run
Run the full DPO training command for 400 iterations with a batch size of 2, a learning rate of 5e-5, a beta of 0.1, reporting every 25 steps, evaluating every 50 steps, saving the adapter to the DPO adapters folder, and piping all output to a log file.

**Verify:** The DPO adapter folder contains adapter files. Open the log and confirm the reward margin (the gap between chosen and rejected log-probabilities) trends positive over the course of training. Expect 20–40 minutes on M1.

---

### 5.3 — Smoke test the DPO adapter
Write a short script that loads the base model with the DPO adapter and generates a response to a simple email prompt.

**Verify (manual):** Compare the output to the SFT smoke test from 2.3. The phrasing should feel noticeably less corporate. Sentence lengths should vary more. No slop phrases should appear.

---

## Phase 6 — Post-Training Evaluation [DONE]

### 6.1 — Generate DPO outputs for all 100 eval prompts
Reuse the generation script from 3.2, replacing the SFT adapter path with the DPO adapter path and the output filename with a new DPO-specific filename in the evals folder.

**Verify:** The new output file has exactly 100 lines.

---

### 6.2 — Score DPO outputs with the same metrics
Reuse the scoring script from 3.3, pointing it at the DPO outputs file and saving results to a new DPO-specific summary JSON file.

**Verify:** The summary JSON exists.

---

### 6.3 — Print a comparison table
Write a script that reads both summary JSON files (SFT baseline and DPO final) and prints a formatted table showing the SFT value, DPO value, and delta for each of the three metrics: mean slop count, mean burstiness, and mean word count.

**Verify:** Mean slop count should be lower after DPO. Mean burstiness should be higher. If both moved in the wrong direction, the DPO pair quality in Phase 4 needs review — pairs may not have had enough stylistic contrast between chosen and rejected.

---

### 6.4 — Evaluate Passive Voice Rate
Write a script using an NLP tool (like spaCy or nltk) to analyze the outputs for passive voice usage and compare the base model vs. the final model.

**Verify:** The final model should exhibit a lower passive voice rate than the baseline model, reflecting a more direct and active writing style.

---


