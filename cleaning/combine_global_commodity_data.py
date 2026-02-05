import pandas as pd
import os
from cmo import clean_worldbank_commodity_data  # ✅ Import from cmo.py

# === 1️⃣ Load World Bank data (using function from cmo.py) ===
def load_worldbank_data(file_path):
    worldbank_df = clean_worldbank_commodity_data(file_path)
    worldbank_df["Source"] = "World Bank"
    print("World Bank data (from cmo.py):")
    print(worldbank_df.head())
    return worldbank_df


# === 2️⃣ Load IMF commodity data ===
def load_imf_data(file_path):
    imf_df = pd.read_csv(file_path)
    imf_df.columns = imf_df.columns.str.strip()
    imf_df.rename(columns={
        "Commodity": "Commodity",
        "Date": "Year",
        "Price": "Value"
    }, inplace=True)
    imf_df["Source"] = "IMF"
    print("IMF data:")
    print(imf_df.head())
    return imf_df


# === 3️⃣ Load FAO data ===
def load_fao_data(file_path):
    fao_df = pd.read_excel(file_path)
    fao_df.columns = fao_df.columns.astype(str).str.strip()
    fao_df = fao_df.melt(id_vars=[fao_df.columns[0]], var_name="Year", value_name="Value")
    fao_df["Source"] = "FAO"
    fao_df.rename(columns={fao_df.columns[0]: "Commodity"}, inplace=True)
    #print("FAO data:")
    #print(df.head())
    return fao_df


# === 4️⃣ Load Retail/Wholesale data ===
def load_retail_wholesale_data(file_path):
    retail_df = pd.read_excel(file_path)
    retail_df.columns = retail_df.columns.astype(str).str.strip()
    retail_df.rename(columns={
        retail_df.columns[0]: "Commodity",
        retail_df.columns[1]: "Year",
        retail_df.columns[-1]: "Value"
    }, inplace=True)
    retail_df["Source"] = "Retail_Wholesale"
   # print("Retail/Wholesale data:")
   # print(df.head()
    return retail_df


# === 5️⃣ Combine all datasets ===
def combine_all_data(paths):
    worldbank_df = load_worldbank_data(paths["worldbank"])
    imf_df = load_imf_data(paths["imf"])
    fao_df = load_fao_data(paths["fao"])
    retail_df = load_retail_wholesale_data(paths["retail"])

    print("Shapes before merge:")
    print("World Bank:", worldbank_df.shape)
    print("IMF:", imf_df.shape)
    print("FAO:", fao_df.shape)
    print("Retail:", retail_df.shape)

    combined = pd.concat([worldbank_df, imf_df, fao_df, retail_df], ignore_index=True)
    combined["Year"] = combined["Year"].astype(str).str.extract(r"(\d{4})")
    combined.dropna(subset=["Year", "Value"], inplace=True)

    print("✅ Combined Dataset Shape:", combined.shape)
    print("📊 Sample data:")
    print(combined.head(10))

    combined.to_csv("global_commodity_data_merged.csv", index=False)
    print("💾 Saved to global_commodity_data_merged.csv")

    return combined


# === 6️⃣ Define file paths ===
paths = {
    "worldbank": "/Users/malini/Documents/EchoSynth/data_eco/CMO-Data-Annual.xlsx",
    "imf": "/Users/malini/Documents/EchoSynth/data_eco/commodity_data_combined.csv",
    "fao": "/Users/malini/Documents/EchoSynth/data_eco/food_price_indices_data_oct.xls",
    "retail": "/Users/malini/Documents/EchoSynth/data_eco/International_Retail and Wholesale_Wed_Oct_29_2025.xlsx"
}

# === 7️⃣ Run the pipeline ===
if __name__ == "__main__":
    combined_data = combine_all_data(paths)
