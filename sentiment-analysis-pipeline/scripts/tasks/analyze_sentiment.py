import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

def main():
    print("=" * 60)
    print("TASK 3: SENTIMENT ANALYSIS")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/cleaned_messages.csv')
    sia = SentimentIntensityAnalyzer()
    
    results = []
    for _, row in df.iterrows():
        scores = sia.polarity_scores(row['cleaned_text'])
        sentiment = 'positive' if scores['compound'] > 0.05 else ('negative' if scores['compound'] < -0.05 else 'neutral')
        results.append({**row, 'vader_compound': scores['compound'], 'final_sentiment': sentiment})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/processed/sentiment_scores.csv', index=False)
    
    print(f"✅ Analyzed {len(results_df)} messages")
    print(results_df['final_sentiment'].value_counts())
    return True

if __name__ == "__main__":
    main()
