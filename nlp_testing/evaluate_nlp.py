import json
import spacy
from transformers import pipeline

# Load SpaCy model for English syntax
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load RoBERTa AI Detector pipeline
print("Loading RoBERTa AI detector model...")
ai_detector = pipeline("text-classification", model="roberta-base-openai-detector", truncation=True, max_length=512)

def calculate_passive_voice_rate(text):
    """
    Returns the number of passive voice constructions per 100 words.
    Uses dependency tags 'auxpass' and 'nsubjpass'.
    """
    doc = nlp(text)
    passive_count = 0
    words_count = 0
    
    for token in doc:
        if not token.is_punct and not token.is_space:
            words_count += 1
        if token.dep_ == "auxpass" or token.dep_ == "nsubjpass":
            passive_count += 1
            
    if words_count == 0:
        return 0
    
    # We count passive instances per 100 words for normalized comparison
    return (passive_count / words_count) * 100

def check_ai_probability(text):
    """
    Returns the probability (0-100) that the text is Fake (AI-generated).
    """
    if not text.strip():
        return 0.0
    
    result = ai_detector(text)[0]
    
    if result['label'] == 'Fake':
        return result['score'] * 100
    else:
        # If it says 'Real', the Fake probability is 100 - Real probability
        return (1.0 - result['score']) * 100

def evaluate_file(filepath):
    print(f"\\nEvaluating {filepath}...")
    passive_rates = []
    fake_probs = []
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # Check which key holds the response based on the file type
                resp = item.get("base_model_response", item.get("dpo_response", ""))
                
                if not resp: continue
                
                p_rate = calculate_passive_voice_rate(resp)
                f_prob = check_ai_probability(resp)
                
                passive_rates.append(p_rate)
                fake_probs.append(f_prob)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
        
    avg_passive = sum(passive_rates) / len(passive_rates) if passive_rates else 0
    avg_fake = sum(fake_probs) / len(fake_probs) if fake_probs else 0
    
    print(f"Average Passive Voice Constructions (per 100 words): {avg_passive:.2f}")
    print(f"Average AI-Generated Probability ('Fake' %): {avg_fake:.2f}%")
    
    return {
        "passive_voice_rate": avg_passive,
        "ai_probability": avg_fake
    }

if __name__ == "__main__":
    base_file = "eval_outputs/base_eval_outputs.jsonl"
    dpo_file = "eval_outputs/final_dpo_eval_outputs.jsonl"
    
    print("=== BASE MODEL EVALUATION ===")
    base_results = evaluate_file(base_file)
    
    print("\\n=== DPO MODEL EVALUATION ===")
    dpo_results = evaluate_file(dpo_file)
    
    if base_results and dpo_results:
        print("\\n=== COMPARISON ===")
        passive_diff = dpo_results['passive_voice_rate'] - base_results['passive_voice_rate']
        ai_diff = dpo_results['ai_probability'] - base_results['ai_probability']
        
        print(f"Passive Voice Delta: {passive_diff:+.2f} per 100 words")
        print(f"AI Probability Delta: {ai_diff:+.2f}%")
        
        if passive_diff < 0 and ai_diff < 0:
            print("\\n✅ SUCCESS: The DPO model uses LESS passive voice and scores lower on the AI detector! It is more human-like.")
        elif passive_diff > 0 or ai_diff > 0:
            print("\\n⚠️ WARNING: The DPO model performed worse on at least one metric. We may need to reconsider.")
