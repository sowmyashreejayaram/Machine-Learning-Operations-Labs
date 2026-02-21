import pandas as pd
import yaml
import random
import json
import os
from datetime import datetime, timedelta

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

POSITIVE = [
    "I absolutely love this! 😊",
    "Best product ever! ⭐⭐⭐⭐⭐",
    "Amazing quality! Highly recommend! 👍",
]

NEGATIVE = [
    "Very disappointed 😞",
    "Terrible quality! Don't buy! 👎",
    "Complete waste of money 😠",
]

NEUTRAL = [
    "It's okay, nothing special.",
    "Pretty average overall.",
    "Does what it says, that's all.",
]

def main():
    print("=" * 60)
    print("TASK 1: GENERATING DATA")
    print("=" * 60)
    
    config = load_config()
    messages = []
    
    for i in range(config['data_generation']['num_messages']):
        if i % 10 < 4:
            text = random.choice(POSITIVE)
            sentiment = 'positive'
        elif i % 10 < 7:
            text = random.choice(NEUTRAL)
            sentiment = 'neutral'
        else:
            text = random.choice(NEGATIVE)
            sentiment = 'negative'
        
        messages.append({
            'id': f'msg_{i:05d}',
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'true_sentiment': sentiment
        })
    
    df = pd.DataFrame(messages)
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/raw_messages.csv', index=False)
    
    print(f"✅ Generated {len(messages)} messages")
    print(f"Saved to: data/raw/raw_messages.csv")
    return True

if __name__ == "__main__":
    main()
