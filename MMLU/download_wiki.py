import json
import os
from datasets import load_dataset

def main():
    OUTPUT_FILE = "background_wiki.json"
    
    print("Downloading wikitext-2-raw-v1...")
    # Using 'raw' to get raw text, which we will filter manually
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    
    print(f"Total entries: {len(ds)}")
    
    # Filter for non-empty, decent length paragraphs to avoid headers/empty lines
    # > 50 chars seems reasonable for a "background document"
    valid_texts = []
    
    for item in ds:
        text = item['text'].strip()
        if len(text) > 50:
            valid_texts.append(text)
            
        if len(valid_texts) >= 1000:
            break
            
    print(f"Selected {len(valid_texts)} generic background texts.")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(valid_texts, f, indent=2)
        
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
