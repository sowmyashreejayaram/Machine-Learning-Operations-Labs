import pandas as pd
import yaml
import os

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 60)
    print("TASK 2: PREPROCESSING")
    print("=" * 60)
    
    df = pd.read_csv('data/raw/raw_messages.csv')
    df['cleaned_text'] = df['text'].str.lower()
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/cleaned_messages.csv', index=False)
    
    print(f"✅ Preprocessed {len(df)} messages")
    return True

if __name__ == "__main__":
    main()
