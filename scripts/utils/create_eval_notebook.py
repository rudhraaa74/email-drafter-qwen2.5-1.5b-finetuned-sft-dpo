import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Introduction
cell1 = nbf.v4.new_markdown_cell("# DPO Training Evaluation: Base Model vs. Final DPO Model\nThis notebook analyzes the evaluation outputs and visualizes the improvements in email generation quality after DPO alignment.")

# Cell 1.5: Install Dependencies
cell_install = nbf.v4.new_code_cell("!pip install -q matplotlib seaborn pandas numpy")

# Cell 2: Imports
cell2 = nbf.v4.new_code_cell("""import json
import re
import string
import statistics
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set seaborn style for better visuals
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)""")

# Cell 3: Metrics Logic (from evaluate.py)
cell3 = nbf.v4.new_code_cell("""SLOP_PHRASES = [
    "hope this email finds you well", "hope this finds you well", "i hope you are doing well",
    "i am writing to", "just writing to", "please find attached", "do not hesitate to",
    "feel free to reach out", "let me know if you have any questions", "thank you for your time",
    "thank you for your attention", "best regards", "kind regards", "sincerely", "delve",
    "synergy", "leverage", "testament", "undeniable", "crucial", "essential",
    "it is important to note", "additionally,"
]

HEDGE_WORDS = ["would", "could", "might", "greatly", "kindly", "appreciate", "hope", "perhaps"]

STOP_WORDS = {
    "the", "to", "and", "a", "of", "in", "i", "is", "for", "that", "you", "it",
    "on", "with", "as", "this", "be", "are", "have", "not", "at", "we", "or",
    "if", "your", "can", "will", "my", "please", "any", "from", "by", "an", "am",
    "name", "subject"
}

def calculate_metrics(text):
    text_lower = text.lower()
    
    slop_count = sum(text_lower.count(p) for p in SLOP_PHRASES)
    hedge_count = sum(len(re.findall(rf'\\b{hw}\\b', text_lower)) for hw in HEDGE_WORDS)
    
    words = text.split()
    length = len(words)
    unique_word_ratio = len(set(words)) / length if length > 0 else 0.0
    
    single_word_counts = Counter()
    clean_words = [w.strip(string.punctuation) for w in text_lower.split()]
    for w in clean_words:
        if w and w not in STOP_WORDS:
            single_word_counts[w] += 1
            
    sentences = [s.strip() for s in re.split(r'[.!?\\n]', text) if len(s.strip()) > 0]
    
    opening_slop = 1 if len(sentences) > 0 and any(p in " ".join(sentences[:3]).lower() for p in SLOP_PHRASES) else 0
    closing_slop = 1 if len(sentences) > 0 and any(p in " ".join(sentences[-3:]).lower() for p in SLOP_PHRASES) else 0
        
    return {
        "slop_count": slop_count,
        "hedge_count": hedge_count,
        "length": length,
        "unique_word_ratio": unique_word_ratio,
        "opening_slop_flag": opening_slop,
        "closing_slop_flag": closing_slop,
        "single_word_counts": single_word_counts
    }""")

# Cell 4: Load and Process Data
cell4 = nbf.v4.new_code_cell("""def process_file(filepath, response_key):
    results = []
    word_counts = Counter()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                resp = item.get(response_key, "")
                if not resp: continue
                m = calculate_metrics(resp)
                results.append(m)
                word_counts.update(m["single_word_counts"])
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None, None
        
    df = pd.DataFrame(results)
    return df, word_counts

base_df, base_words = process_file('eval_outputs/base_eval_outputs.jsonl', 'base_model_response')
dpo_df, dpo_words = process_file('eval_outputs/final_dpo_eval_outputs.jsonl', 'base_model_response') # The script saved it under 'base_model_response' for compatibility

print(f"Loaded {len(base_df)} base emails and {len(dpo_df)} DPO emails.")""")

# Cell 5: Compare High-Level Metrics
cell5 = nbf.v4.new_markdown_cell("## 📊 Comparison of Key Metrics")
cell6 = nbf.v4.new_code_cell("""summary = pd.DataFrame({
    'Metric': [
        'Avg Slop Phrases', 
        'Avg Hedge Words', 
        'Avg Length (words)', 
        'Unique Word Ratio', 
        'Opening Slop %', 
        'Closing Slop %'
    ],
    'Base Model': [
        base_df['slop_count'].mean(),
        base_df['hedge_count'].mean(),
        base_df['length'].mean(),
        base_df['unique_word_ratio'].mean(),
        base_df['opening_slop_flag'].mean() * 100,
        base_df['closing_slop_flag'].mean() * 100
    ],
    'DPO Model': [
        dpo_df['slop_count'].mean(),
        dpo_df['hedge_count'].mean(),
        dpo_df['length'].mean(),
        dpo_df['unique_word_ratio'].mean(),
        dpo_df['opening_slop_flag'].mean() * 100,
        dpo_df['closing_slop_flag'].mean() * 100
    ]
})
display(summary.round(2))""")

# Cell 7: Plotting the Improvements
cell7 = nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Slop Count
sns.barplot(x=['Base', 'DPO'], y=[summary.iloc[0]['Base Model'], summary.iloc[0]['DPO Model']], ax=axes[0, 0], palette=['#ff9999', '#66b3ff'])
axes[0, 0].set_title('Average Slop Phrases per Email', fontsize=14)
axes[0, 0].set_ylabel('Count')

# 2. Length
sns.barplot(x=['Base', 'DPO'], y=[summary.iloc[2]['Base Model'], summary.iloc[2]['DPO Model']], ax=axes[0, 1], palette=['#ff9999', '#66b3ff'])
axes[0, 1].set_title('Average Email Length (words)', fontsize=14)
axes[0, 1].set_ylabel('Words')

# 3. Hedge Words
sns.barplot(x=['Base', 'DPO'], y=[summary.iloc[1]['Base Model'], summary.iloc[1]['DPO Model']], ax=axes[1, 0], palette=['#ffcc99', '#99ff99'])
axes[1, 0].set_title('Average Hedge Words per Email', fontsize=14)
axes[1, 0].set_ylabel('Count')

# 4. Opening/Closing Slop
slop_pos = pd.DataFrame({
    'Model': ['Base', 'Base', 'DPO', 'DPO'],
    'Type': ['Opening', 'Closing', 'Opening', 'Closing'],
    'Percentage': [summary.iloc[4]['Base Model'], summary.iloc[5]['Base Model'], summary.iloc[4]['DPO Model'], summary.iloc[5]['DPO Model']]
})
sns.barplot(data=slop_pos, x='Type', y='Percentage', hue='Model', ax=axes[1, 1], palette=['#ff9999', '#66b3ff'])
axes[1, 1].set_title('% of Emails with Slop at Start/End', fontsize=14)
axes[1, 1].set_ylabel('Percentage (%)')

plt.tight_layout()
plt.show()""")

# Cell 8: Vocabulary Shift
cell8 = nbf.v4.new_markdown_cell("## 🔠 Vocabulary Shift (Top 20 Words)")
cell9 = nbf.v4.new_code_cell("""# Get top 20 words
base_top_20 = dict(base_words.most_common(20))
dpo_top_20 = dict(dpo_words.most_common(20))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(x=list(base_top_20.values()), y=list(base_top_20.keys()), ax=axes[0], color='#ff9999')
axes[0].set_title('Base Model: Top 20 Words', fontsize=14)
axes[0].set_xlabel('Frequency')

sns.barplot(x=list(dpo_top_20.values()), y=list(dpo_top_20.keys()), ax=axes[1], color='#66b3ff')
axes[1].set_title('DPO Model: Top 20 Words', fontsize=14)
axes[1].set_xlabel('Frequency')

plt.tight_layout()
plt.show()""")

nb['cells'] = [cell1, cell_install, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

# Write out the notebook in the main directory
with open('/Users/rudhrakoul/anti-ai-email-alignment/Evaluation_Breakdown.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook Evaluation_Breakdown.ipynb created successfully!")
