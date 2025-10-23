"""
data_ingestion.py
-----------------
Module for downloading, reading, cleaning, and preparing the Amazon Reviews dataset
for use in the EchoSynth project.
"""

import os
import pandas as pd
import re
import gzip
import json
from tqdm import tqdm

# ============== CONFIG ==============
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "amazon_reviews.json.gz")
PROCESSED_FILE = os.path.join(DATA_DIR, "cleaned_reviews.csv")
SAMPLE_SIZE = 100000  # adjust based on your system (e.g., 100k rows for dev)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


# ============== TEXT CLEANING HELPERS ==============
def clean_text(text: str) -> str:
    """Basic text normalization."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============== DATA LOADING FUNCTIONS ==============
def load_gzip_json(filepath: str, sample_size: int = 100000) -> pd.DataFrame:
    """
    Reads a compressed .json.gz file containing line-delimited JSON (Amazon review format).
    Returns a sampled DataFrame.
    """
    print(f"🔹 Loading dataset from: {filepath}")

    reviews = []
    with gzip.open(filepath, "rb") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            reviews.append(json.loads(line))
            if (i + 1) % 10000 == 0:
                print(f"   → Loaded {i+1:,} records...")

    df = pd.DataFrame(reviews)
    print(f"✅ Loaded {len(df):,} rows.")
    return df


# ============== CLEANING PIPELINE ==============
def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Select and clean relevant fields from the dataset."""
    print("🧹 Cleaning and normalizing data...")

    # Keep key columns if they exist
    keep_cols = ["review_body", "product_title", "star_rating", "product_category"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Rename for consistency
    df.rename(columns={
        "review_body": "review",
        "star_rating": "rating",
        "product_title": "product",
        "product_category": "category"
    }, inplace=True)

    # Clean text
    tqdm.pandas(desc="Cleaning reviews")
    df["review"] = df["review"].progress_apply(clean_text)

    # Drop missing or short entries
    df.dropna(subset=["review"], inplace=True)
    df = df[df["review"].str.len() > 10]

    print(f"✅ Cleaned dataset: {len(df):,} valid reviews.")
    return df


# ============== MAIN EXECUTION ==============
def main():
    """Main entry point for data ingestion."""
    print("🚀 Starting data ingestion for EchoSynth...")

    # Check if file exists
    if not os.path.exists(RAW_FILE):
        print(f"❌ Raw data not found at {RAW_FILE}")
        print("➡️  Please download the Amazon Reviews dataset from:")
        print("   https://registry.opendata.aws/amazon-reviews/")
        print("   and place it under the 'data/' directory.")
        return

    # Load and preprocess
    df = load_gzip_json(RAW_FILE, sample_size=SAMPLE_SIZE)
    df_clean = preprocess_reviews(df)

    # Save processed data
    df_clean.to_csv(PROCESSED_FILE, index=False)
    print(f"💾 Saved cleaned data to: {PROCESSED_FILE}")


if __name__ == "__main__":
    main()
