from collections import Counter
import json
import argparse
import re
import statistics
import os
import string

# Define our list of banned "slop" phrases
SLOP_PHRASES = [
    "hope this email finds you well",
    "hope this finds you well",
    "i hope you are doing well",
    "i am writing to",
    "just writing to",
    "please find attached",
    "do not hesitate to",
    "feel free to reach out",
    "let me know if you have any questions",
    "thank you for your time",
    "thank you for your attention",
    "best regards",
    "kind regards",
    "sincerely",
    "delve",
    "synergy",
    "leverage",
    "testament",
    "undeniable",
    "crucial",
    "essential",
    "it is important to note",
    "additionally,"
]

# Define hedge words
HEDGE_WORDS = [
    "would",
    "could",
    "might",
    "greatly",
    "kindly",
    "appreciate",
    "hope",
    "perhaps"
]

# Basic stop words to exclude from top single words list
STOP_WORDS = {
    "the", "to", "and", "a", "of", "in", "i", "is", "for", "that", "you", "it",
    "on", "with", "as", "this", "be", "are", "have", "not", "at", "we", "or",
    "if", "your", "can", "will", "my", "please", "any", "from", "by", "an", "am",
    "name", "subject"
}

def calculate_metrics(text):
    text_lower = text.lower()
    
    # 1. Slop Count
    phrase_counts = Counter()
    slop_count = 0
    for phrase in SLOP_PHRASES:
        count = text_lower.count(phrase)
        if count > 0:
            phrase_counts[phrase] += count
            slop_count += count
            
    # 1.5 Hedge Count
    hedge_counts = Counter()
    hedge_count = 0
    for hw in HEDGE_WORDS:
        matches = len(re.findall(rf'\b{hw}\b', text_lower))
        if matches > 0:
            hedge_counts[hw] += matches
            hedge_count += matches
    
    # 2. Length (word count)
    words = text.split()
    length = len(words)
    
    # Unique word ratio
    if length > 0:
        unique_word_ratio = len(set(words)) / length
    else:
        unique_word_ratio = 0.0
        
    # Single Word Frequencies (excluding stop words and punctuation)
    single_word_counts = Counter()
    clean_words = [w.strip(string.punctuation) for w in text_lower.split()]
    for w in clean_words:
        if w and w not in STOP_WORDS:
            single_word_counts[w] += 1
    
    # 3. Burstiness (variation in sentence lengths)
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 0]
    sentence_lengths = [len(s.split()) for s in sentences]
    
    if len(sentence_lengths) > 1:
        burstiness_stdev = statistics.stdev(sentence_lengths)
        burstiness_var = statistics.variance(sentence_lengths)
    else:
        burstiness_stdev = 0.0
        burstiness_var = 0.0
        
    # 4. Opening and Closing Slop Flags
    opening_slop_flag = 0
    closing_slop_flag = 0
    
    if len(sentences) > 0:
        first_few = " ".join(sentences[:3]).lower()
        if any(phrase in first_few for phrase in SLOP_PHRASES):
            opening_slop_flag = 1
            
        last_few = " ".join(sentences[-3:]).lower()
        if any(phrase in last_few for phrase in SLOP_PHRASES):
            closing_slop_flag = 1
        
    return {
        "slop_count": slop_count,
        "hedge_count": hedge_count,
        "phrase_counts": phrase_counts,
        "hedge_word_counts": hedge_counts,
        "single_word_counts": single_word_counts,
        "length": length,
        "unique_word_ratio": unique_word_ratio,
        "burstiness_stdev": round(burstiness_stdev, 2),
        "burstiness_var": round(burstiness_var, 2),
        "sentence_count": len(sentences),
        "opening_slop_flag": opening_slop_flag,
        "closing_slop_flag": closing_slop_flag
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate model outputs against comprehensive metrics.")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file to evaluate")
    parser.add_argument("--response_key", type=str, default="base_model_response", help="JSON key containing the generated text")
    args = parser.parse_args()
    
    metrics_sum = {
        "slop_count": 0,
        "hedge_count": 0,
        "length": 0,
        "unique_word_ratio": 0.0,
        "burstiness_stdev": 0.0,
        "burstiness_var": 0.0,
        "sentence_count": 0,
        "opening_slop_flag": 0,
        "closing_slop_flag": 0
    }
    
    count = 0
    all_phrase_counts = Counter()
    all_hedge_counts = Counter()
    all_single_word_counts = Counter()
    
    with open(args.input, 'r') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            response = item.get(args.response_key, "")
            if not response:
                continue
                
            m = calculate_metrics(response)
            
            for k in metrics_sum:
                metrics_sum[k] += m[k]
                
            all_phrase_counts.update(m["phrase_counts"])
            all_hedge_counts.update(m["hedge_word_counts"])
            all_single_word_counts.update(m["single_word_counts"])
            count += 1
            
    if count == 0:
        print("No valid responses found to evaluate.")
        return
        
    report = []
    report.append(f"--- Comprehensive Evaluation Results for {args.input} ---")
    report.append(f"Total evaluated: {count} emails")
    report.append(f"Average Slop Count:       {metrics_sum['slop_count'] / count:.2f} phrases/email")
    report.append(f"Average Hedge Count:      {metrics_sum['hedge_count'] / count:.2f} words/email")
    report.append(f"Opening Slop Ratio:       {metrics_sum['opening_slop_flag'] / count * 100:.1f}% of emails start with slop")
    report.append(f"Closing Slop Ratio:       {metrics_sum['closing_slop_flag'] / count * 100:.1f}% of emails end with slop")
    report.append(f"Average Length:           {metrics_sum['length'] / count:.2f} words")
    report.append(f"Average Sentences:        {metrics_sum['sentence_count'] / count:.2f} sentences/email")
    report.append(f"Sentence Length Std Dev:  {metrics_sum['burstiness_stdev'] / count:.2f}")
    report.append(f"Sentence Length Variance: {metrics_sum['burstiness_var'] / count:.2f}")
    report.append(f"Unique Word Ratio:        {metrics_sum['unique_word_ratio'] / count:.3f} (higher is better/more vocabulary)")
    
    report.append("\n--- Slop Phrase Breakdown ---")
    if not all_phrase_counts:
        report.append("No slop phrases found!")
    else:
        for phrase, freq in all_phrase_counts.most_common():
            report.append(f"'{phrase}': {freq} times")
            
    report.append("\n--- Hedge Word Breakdown ---")
    if not all_hedge_counts:
        report.append("No hedge words found!")
    else:
        for word, freq in all_hedge_counts.most_common():
            report.append(f"'{word}': {freq} times")
            
    report.append("\n--- Top 40 Single Words (excluding basic stop words) ---")
    if not all_single_word_counts:
        report.append("No words found!")
    else:
        for word, freq in all_single_word_counts.most_common(40):
            report.append(f"'{word}': {freq} times")
            
    report_text = "\n".join(report)
    print(report_text)
    
    base_name = os.path.basename(args.input).replace(".jsonl", "")
    out_dir = os.path.dirname(args.input)
    breakdown_file = os.path.join(out_dir, f"{base_name}_comprehensive_report.txt")
    
    with open(breakdown_file, "w") as f:
        f.write(report_text + "\n")
        
    print(f"\nDetailed comprehensive report saved to: {breakdown_file}")

if __name__ == "__main__":
    main()
