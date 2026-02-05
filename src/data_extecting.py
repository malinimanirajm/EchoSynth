import pandas as pd
import glob
import os



df = pd.read_excel("data_eco/CMO-Historical-Data-Annual.xlsx", engine="openpyxl") 

# 📂 Path to your Excel files
folder_path = "/Users/malini/Documents/EchoSynth/data_eco"

# 🧾 Get all Excel files (sorted alphabetically by default)
excel_files = sorted(glob.glob(os.path.join(folder_path, "*.xls*")))

print("Found Excel files:")
for i, f in enumerate(excel_files, 1):
    print(f"{i}. {os.path.basename(f)}")

# ⚙️ Load each file
dataframes = [pd.read_excel(f) for f in excel_files]

# ✅ Combine logic
if len(dataframes) >= 4:
    # Combine first two → Commodity / Price data
    commodity_data = pd.concat(dataframes[:2], ignore_index=True)

    # Combine next two → Global Risk / Geopolitical data
    risk_data = pd.concat(dataframes[2:4], ignore_index=True)

    print("\n✅ Commodity Data Shape:", commodity_data.shape)
    print("✅ Risk Data Shape:", risk_data.shape)

    # Optional preview
    print("\n📊 Commodity Data Preview:")
    print(commodity_data.columns.to_list())

    print("\n🌍 Risk Data Preview:")
    print(risk_data.columns.to_list())

    # 💾 Save to CSV for future stages
    commodity_data.to_csv(os.path.join(folder_path, "commodity_data_combined.csv"), index=False)
    risk_data.to_csv(os.path.join(folder_path, "global_risk_data_combined.csv"), index=False)

    print("\n💾 Saved combined files:")
    print(" - commodity_data_combined.csv")
    print(" - global_risk_data_combined.csv")

else:
    print("❌ Need at least 4 Excel files in the folder to perform this operation.")
