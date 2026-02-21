import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    print("=" * 60)
    print("TASK 4: CREATING VISUALIZATIONS")
    print("=" * 60)
    
    df = pd.read_csv('data/processed/sentiment_scores.csv')
    
    os.makedirs('visualizations', exist_ok=True)
    
    # Pie chart
    sentiment_counts = df['final_sentiment'].value_counts()
    plt.figure(figsize=(10, 8))
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
    plt.title('Sentiment Distribution')
    plt.savefig('visualizations/sentiment_distribution.png', dpi=300)
    plt.close()
    
    print("✅ Created visualization: sentiment_distribution.png")
    return True

if __name__ == "__main__":
    main()
