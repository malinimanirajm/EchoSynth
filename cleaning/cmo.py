import pandas as pd
import os

def clean_worldbank_commodity_data(file_path):
    """
    Cleans the World Bank 'CMO-Historical-Data-Annual.xlsx' file
    by detecting header rows, removing metadata, and keeping only useful columns.
    Works for .xlsx files.
    """
    # 🧠 Step 0: Verify file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File not found: {file_path}")

    # 🧩 Load Excel file dynamically using provided file_path
    xl = pd.ExcelFile(file_path)
    print("📄 Available sheets:", xl.sheet_names)

    sheet = "Annual Prices (Real)"   # You can also switch to "Annual Indices (Real)"

    # Step 1: Read without header to detect where the actual data starts
    df_raw = pd.read_excel(file_path, sheet_name=sheet, header=None)

    # Step 2: Detect first row that contains numeric years (e.g., 1960–2025)
    start_row = df_raw[
        df_raw.apply(lambda x: x.astype(str).str.contains(r"19\d{2}|20\d{2}", regex=True).any(), axis=1)
    ].index[0]

    print(f"🔍 Detected header row at index: {start_row}")

    # Step 3: Re-read with correct header
    df = pd.read_excel(file_path, sheet_name=sheet, header=start_row)

    # Step 4: Remove unnecessary columns
    df = df.loc[:, ~df.columns.astype(str).str.contains("Unnamed", case=False)]
    df = df.dropna(subset=[df.columns[0]])  # Drop blank commodity names

    # Step 5: Clean column names
    df.columns = df.columns.astype(str).str.strip()

    print("✅ Cleaned DataFrame shape:", df.shape)
    print("\n📊 Sample data:")
    print(df.head(5))

    return df


# ✅ Example usage
file_path = "/Users/malini/Documents/EchoSynth/data_eco/CMO-Data-Annual.xlsx"
cleaned_worldbank_df = clean_worldbank_commodity_data(file_path)
