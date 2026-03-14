"""
Download Wine Quality dataset from UCI Machine Learning Repository.
"""

import os
import pandas as pd


def download_wine_data():
    """Download and save the Wine Quality dataset."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, "winequality-red.csv")
    
    if os.path.exists(output_path):
        print(f"Data already exists at {output_path}")
        return output_path
    
    print(f"Downloading Wine Quality dataset...")
    df = pd.read_csv(url, sep=";")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Quality distribution:\n{df['quality'].value_counts().sort_index()}")
    
    return output_path


if __name__ == "__main__":
    download_wine_data()
