import json
import random

def is_header(line):
    headers = ['To:', 'From:', 'Date:', 'Subject:', 'CC:', 'Cc:', 'Bcc:', 'Sent:']
    for h in headers:
        if line.startswith(h):
            return True
    if line.startswith('X-'):
        return True
    return False

def is_forward_or_reply(line):
    lower_line = line.lower()
    return 'forwarded by' in lower_line or 'original message' in lower_line

def is_legal_boilerplate(line):
    lower_line = line.lower()
    phrases = [
        'confidentiality notice',
        'this email and any attachments are confidential',
        'intended solely for the use of the individual',
        'if you have received this email in error',
        'this message is for the sole use',
        'strictly prohibited'
    ]
    for p in phrases:
        if p in lower_line:
            return True
    return False

def clean_email(raw_body):
    lines = raw_body.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Stop at forwarded/replied messages
        if is_forward_or_reply(stripped):
            break
            
        # Stop at legal boilerplate
        if is_legal_boilerplate(stripped):
            break
            
        # Skip headers
        if is_header(stripped):
            continue
            
        cleaned_lines.append(line)
        
    cleaned_body = '\n'.join(cleaned_lines).strip()
    return cleaned_body

def main():
    input_path = "raw_data/aeslc_raw.jsonl"
    output_path = "raw_data/aeslc_cleaned.jsonl"
    
    valid_emails = []
    discarded = 0
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        for row_str in fin:
            row = json.loads(row_str)
            raw_body = row.get('email_body', '')
            
            cleaned_body = clean_email(raw_body)
            word_count = len(cleaned_body.split())
            
            if 15 <= word_count <= 300:
                valid_emails.append(cleaned_body)
            else:
                discarded += 1
                
    random.seed(42)
    sampled_emails = random.sample(valid_emails, min(2000, len(valid_emails)))
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for email in sampled_emails:
            fout.write(json.dumps({"email": email}) + "\n")
                
    print(f"Cleaning complete.")
    print(f"Valid emails found: {len(valid_emails)}")
    print(f"Emails discarded (length filter): {discarded}")
    print(f"Emails saved (capped at 2000): {len(sampled_emails)}")

if __name__ == '__main__':
    main()
