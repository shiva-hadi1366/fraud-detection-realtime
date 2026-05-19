# data_prep.py
# -----------------------------------------
# Data Cleaning & Preprocessing for Fraud Detection
# -----------------------------------------

import pandas as pd
import numpy as np
import os

def load_data(path: str) -> pd.DataFrame:
    """Load raw transaction dataset."""
    df = pd.read_csv(path)
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: remove duplicates, handle missing values."""
    
    # Remove duplicates
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"Removed duplicates: {before - after}")

    # Handle missing values (if any)
    df = df.fillna(0)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and transform existing ones."""

    # Convert timestamp to datetime (if exists)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Example: log-transform amount to reduce skewness
    if "amount" in df.columns:
        df["amount_log"] = np.log1p(df["amount"])

    # Example: normalize time_since_last
    if "time_since_last" in df.columns:
        df["time_since_last_norm"] = (df["time_since_last"] - df["time_since_last"].mean()) / df["time_since_last"].std()

    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numeric using one-hot encoding."""
    
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if len(categorical_cols) > 0:
        print(f"Encoding categorical columns: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    else:
        print("No categorical columns found.")

    return df


def save_processed(df: pd.DataFrame, output_path: str):
    """Save processed dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")


def run_preprocessing():
    """Main pipeline for preprocessing."""
    
    raw_path = "data/raw/transactions.csv"
    processed_path = "data/processed/transactions_processed.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    df = feature_engineering(df)
    df = encode_categorical(df)
    save_processed(df, processed_path)

    print("Data preprocessing completed successfully.")


if __name__ == "__main__":
    run_preprocessing()

