"""
Data preprocessing for Wine Quality classification.
Handles loading, cleaning, feature engineering, and train-test split.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path: str = None) -> pd.DataFrame:
    """Load the Wine Quality dataset."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "winequality-red.csv",
        )
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Preprocess the wine quality data.
    
    Steps:
    1. Separate features and target
    2. Standardize features using StandardScaler
    3. Train-test split
    
    Args:
        df: Raw dataframe
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Separate features and target
    feature_names = [col for col in df.columns if col != "quality"]
    X = df[feature_names].values
    y = df["quality"].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    print(f"Features:  {len(feature_names)}")
    print(f"Classes:   {np.unique(y_train)}")
    
    return X_train, X_test, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)
