# Project History

## Phase 0: Project Setup
- **Environment:** Created Python virtual environment (`venv`).
- **Dependencies:** Installed `mlx-lm`, `datasets`, and confirmed DPO support in `mlx-lm-lora`.
- **Git Repository:** Initialized local git repo, created `.gitignore` (ignoring data, models, artifacts), added remote origin (`git@github.com:rudhraaa74/email-drafter-qwen2.5-1.5b-finetuned-sft-dpo.git`), and pushed the initial commit.
- **Workflow Guidelines:** Created `GEMINI.md` to establish rules for simplicity, surgical changes, and goal-driven execution.

## Phase 1: Data Preparation
- **1.1 & 1.2 — Download Dataset:** Created and ran `scripts/1_download_aeslc.py` to pull the `Yale-LILY/aeslc` dataset from HuggingFace and save it locally to `raw_data/aeslc_raw.jsonl` (14,436 emails).
- **1.3 & 1.4 — Clean and Filter:** Created and ran `scripts/2_clean_aeslc.py`.
  - Removed forwarded messages, replied threads, and legal boilerplate.
  - Stripped metadata headers (To, From, Date, Subject, CC, X- headers).
  - Filtered emails to only keep those between 15 and 300 words.
  - Added a reproducible random sampling step (`random.seed(42)`) to cap the output at exactly 2,000 rows.
  - Saved the final filtered data to `raw_data/aeslc_cleaned.jsonl`.
- **Spot Check:** Created and ran `scripts/3_spot_check.py` to randomly sample 10 emails from the cleaned JSONL and verify that formatting and cleaning logic performed exactly as expected.
- **Git Sync:** Committed all data extraction and cleaning scripts (and `PLAN.md` updates) to the GitHub repository.

---
*Next up: Phase 1.5 - Generating reverse prompts via Kaggle Notebook for the 2,000 cleaned emails.*
