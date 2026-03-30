"""Download Wine Quality Dataset from UCI ML Repository."""
import os
import pandas as pd
import argparse

RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

def download_wine_data(output_path, include_white=False):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Downloading red wine data...")
    red_wine = pd.read_csv(RED_WINE_URL, sep=";")
    red_wine["wine_type"] = "red"
    print(f"  Red wine samples: {len(red_wine)}")
    if include_white:
        print("Downloading white wine data...")
        white_wine = pd.read_csv(WHITE_WINE_URL, sep=";")
        white_wine["wine_type"] = "white"
        print(f"  White wine samples: {len(white_wine)}")
        df = pd.concat([red_wine, white_wine], ignore_index=True)
    else:
        df = red_wine
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} samples to {output_path}")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-white", action="store_true")
    parser.add_argument("--output", default="data/wine_quality.csv")
    args = parser.parse_args()
    download_wine_data(args.output, include_white=args.include_white)
