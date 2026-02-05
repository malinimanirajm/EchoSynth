import pandas as pd
import os

# === 1️⃣ Define dataset paths ===
datasets = {
    "World Bank": "CMO-Data-Annual.xlsx",
    "IMF": "IMF_Commodity_Prices.xlsx",
    "FAO": "FAO_Food_Prices.xlsx",
    "Retail": "Retail_Prices.xlsx",
    "WUI": "World_Uncertainty_Index.xlsx",
    "GPR": "Geopolitical_Risk_Index.xlsx"
}

# === 2️⃣ Helper: load either Excel or CSV safely ===
def safe_load(file_path):
    if not os.path.exists(file_path):
        print(f"🚫 File not found: {file_path}")
        return None
    try:
        if file_path.endswith(".csv"):
            return pd.read_csv(file_path)
        else:
            return pd.read_excel(file_path)
    except Exception as e:
        print(f"🚨 Error reading {file_path}: {e}")
        return None

# === 3️⃣ Detect or create 'Year' column ===
def ensure_year_column(df, source_name):
    df.columns = [str(c).strip() for c in df.columns]

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
        print(f"✅ {source_name}: 'Year' column found. Unique years ({len(df['Year'].unique())}): {sorted(df['Year'].dropna().unique())[:10]}...")
        return df

    # Try to infer from other columns
    possible_date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "month", "time", "period"])]
    if possible_date_cols:
        for col in possible_date_cols:
            try:
                df["Year"] = pd.to_datetime(df[col], errors="coerce").dt.year
                if df["Year"].notna().sum() > 0:
                    print(f"⚙️ {source_name}: Extracted 'Year' from column '{col}'.")
                    return df
            except Exception:
                continue

    # Try to detect year values in column names (for pivoted tables)
    year_like_cols = [c for c in df.columns if any(str(y) in c for y in range(1900, 2030))]
    if year_like_cols:
        df_long = df.melt(id_vars=[df.columns[0]], var_name="Year", value_name="Value")
        df_long["Year"] = pd.to_numeric(df_long["Year"].str.extract(r"(\d{4})")[0], errors='coerce')
        print(f"🧩 {source_name}: Pivoted year columns into 'Year'.")
        return df_long

    print(f"❌ {source_name}: No 'Year' or date-like column found.")
    return df

# === 4️⃣ Standardize all datasets ===
def process_all_datasets(datasets):
    os.makedirs("standardized_data", exist_ok=True)
    standardized = {}

    for name, path in datasets.items():
        print(f"\n📘 Processing: {name}")
        df = safe_load(path)
        if df is not None:
            df_clean = ensure_year_column(df, name)
            standardized[name] = df_clean
            # Save cleaned file for reference
            out_path = f"standardized_data/{name.replace(' ', '_')}_cleaned.csv"
            df_clean.to_csv(out_path, index=False)
            print(f"💾 Saved cleaned dataset: {out_path}")
        else:
            print(f"🚫 Skipping {name} (file not loaded).")

    return standardized

# === 5️⃣ Run the script ===
if __name__ == "__main__":
    all_data = process_all_datasets(datasets)

    # Combine all unique years across datasets
    all_years = set()
    for name, df in all_data.items():
        if "Year" in df.columns:
            all_years.update(df["Year"].dropna().astype(int).tolist())

    print("\n🌍 Combined Unique Years Across All Datasets:")
    print(sorted(all_years))
