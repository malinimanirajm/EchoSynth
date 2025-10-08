# echosynth_data_setup.py

import pandas as pd
from datasets import load_dataset
import re
import os

# -----------------------------
# Step 1: Create data folder
# -----------------------------
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# -----------------------------
# Step 2: Load Amazon Polarity dataset
# -----------------------------
print("Loading Amazon Polarity dataset...")
dataset = load_dataset("amazon_polarity")
df_train = dataset['train'].to_pandas()
df_test = dataset['test'].to_pandas()
print(f"Train set: {df_train.shape}, Test set: {df_test.shape}")

# -----------------------------
# Step 3: Filter for Beauty / Personal Care
# -----------------------------
keywords = ['beauty', 'skincare', 'cosmetic', 'makeup', 'personal care']
pattern = '|'.join(keywords)

df_train_filtered = df_train[df_train['content'].str.contains(pattern, case=False, na=False)]
df_test_filtered = df_test[df_test['content'].str.contains(pattern, case=False, na=False)]

print(f"Filtered train set: {df_train_filtered.shape}, Filtered test set: {df_test_filtered.shape}")

# -----------------------------
# Step 4: Clean text function
# -----------------------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces/newlines
    text = re.sub(r'[^A-Za-z0-9 .,!?\'-]', '', text)  # remove special characters
    return text.strip()

df_train_filtered['content_clean'] = df_train_filtered['content'].apply(clean_text)
df_test_filtered['content_clean'] = df_test_filtered['content'].apply(clean_text)

# -----------------------------
# Step 5: Save cleaned CSV
# -----------------------------
train_csv_path = os.path.join(data_dir, "amazon_beauty_train.csv")
test_csv_path = os.path.join(data_dir, "amazon_beauty_test.csv")

df_train_filtered.to_csv(train_csv_path, index=False)
df_test_filtered.to_csv(test_csv_path, index=False)

print(f"Saved cleaned data to:\n - {train_csv_path}\n - {test_csv_path}")
