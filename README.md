<div align="center">
  <h1>📧 Anti-AI Slop Email Assistant</h1>
  <p><strong>A Qwen 2.5 1.5B model finetuned with SFT & DPO to write emails that actually sound like a human.</strong></p>
  
  [![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
  [![Framework](https://img.shields.io/badge/Framework-MLX%20LM-orange.svg)](https://github.com/ml-explore/mlx)
  [![Alignment](https://img.shields.io/badge/Alignment-SFT%20%2B%20DPO-green.svg)]()
</div>

<br/>

## 🎯 The Problem: "AI Slop"
We all know what an AI-generated email sounds like. It's filled with *"I hope this email finds you well,"* *"Delve,"* *"Tapestry,"* and an unnatural, highly-structured passive voice. It lacks the punchy, direct, and slightly irregular cadence ("burstiness") of real human communication.

This project tackles that problem head-on. We took a base **Qwen 2.5 1.5B** model and aggressively trained it to drop the corporate AI speak, resulting in emails that are direct, concise, and genuinely human-like.

---

## 🧪 Methodology: How We Did It

### 1. Supervised Fine-Tuning (SFT)
We started by filtering the **AESLC (Annotated Enron Subject Line Corpus)** to extract real, human-written corporate emails. We cleaned the dataset, removed metadata, and performed a standard Supervised Fine-Tuning pass to teach the model the basic structure of a concise email.

### 2. Direct Preference Optimization (DPO)
To actively punish "AI Slop," we needed preference pairs. We used **Google's Gemini Pro API** as an adversarial judge:
1. We fed prompts to Gemini and asked it to generate two emails:
   - **The Rejected Email:** Full of classic LLM tropes ("I hope this finds you well", passive voice, overly verbose).
   - **The Chosen Email:** Extremely direct, punchy, active voice, zero fluff.
2. We trained the SFT model using the **MLX DPO (Direct Preference Optimization)** pipeline to actively push its weights away from the slop and toward the concise style.

---

## 📊 Major Findings

### Headline Result
Fine-tuning Qwen2.5-1.5B-Instruct via SFT followed by DPO reduced AI-characteristic writing patterns by 96% (slop phrase frequency) while maintaining semantic relevance and natural sentence-pacing variation — trained end-to-end on Apple Silicon (M1) using mlx-lm-lora.

### Quantitative Results

| Metric | Base Model | SFT+DPO Model | Change |
|--------|------------|---------------|--------|
| Slop phrases / email | 3.13 | 0.11 | 📉 -96% |
| Hedge words / email | 2.28 | 0.31 | 📉 -86% |
| Opening ritual phrases | 61.0% | 5.0% | 📉 -92% |
| Closing ritual phrases | 59.0% | 10.0% | 📉 -83% |
| Average length (words) | 134.5 | 53.0 | 📉 -61% |
| Sentence count / email | 13.5 | 5.95 | 📉 -56% |
| Sentence length variance | 60.09 | 56.54 | 📉 -6% |
| Unique word ratio | 0.717 | 0.837 | 📈 +17% |
| Passive voice / 100 words | 1.24 | 1.05 | 📉 -15% |

### Key Finding 1 — Brevity Was Not a Shortcut
The most aggressive change DPO made was reducing average email length by 61%. This raised an important question: was the model genuinely writing more naturally, or did it learn "shorter = preferred" as a cheap proxy for quality (a known DPO failure mode on small preference datasets)?
To test this, sentence-length variance was tracked alongside length. If the model were simply truncating output, variance should have collapsed sharply alongside length. Instead, variance dropped only 6% while length dropped 61% — meaning the model preserved natural pacing variation even as it became more concise. This is evidence against simple reward-hacking on brevity.

### Key Finding 2 — Style Improvement Is Register-Dependent
Manual qualitative testing on held-out prompts revealed that the stylistic improvement is not uniform across email types. Personal/casual-register prompts (e.g. "ask my landlord to fix a leak") showed strong, natural-sounding improvement. Business/transactional-register prompts (e.g. supplier complaints, service cancellations) showed weaker improvement and occasionally reverted toward stiffer, more templated phrasing.
This is attributed to an imbalance in the DPO preference dataset, which likely skewed toward casual-register source emails during manual ranking. This is documented as a known limitation rather than papered over — and represents a clear next step (collecting more business-register preference pairs) for anyone extending this work.

### Key Finding 3 — Slop Is Concentrated in a Small Phrase Set
Analysis of the base model's failure modes showed that a small number of high-frequency phrases — "best regards" (78 occurrences/100 emails), "hope this email finds you well" (70), and "i am writing to" (41) — accounted for the large majority of detected slop. This suggests AI-characteristic phrasing in email generation is concentrated rather than diffuse, which is why a relatively small (~250 pair) DPO dataset was able to produce a large aggregate effect.

---

## 📂 Repository Structure

* `scripts/data_pipeline/`: Scripts for downloading AESLC, cleaning data, and querying the Gemini API to generate synthetic DPO pairs.
* `scripts/training/`: Shell scripts and Python runners for executing MLX SFT and MLX DPO locally on Apple Silicon.
* `scripts/evaluation/`: Scripts for batch-generating responses and calculating stylistic metrics (slop, burstiness, word count).
* `scripts/interaction/`: CLI scripts to chat with the model interactively.
* `nlp_testing/`: Advanced isolated NLP evaluations (`spaCy`, `RoBERTa`) measuring passive voice and AI detectability.
* `adapters/`: (Untracked) Where the LoRA weights for SFT and DPO are stored.
* `notebooks/`: Visual breakdowns of our evaluation metrics.

---


