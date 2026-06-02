import pandas as pd

csv_path = "./metadata_JPSS-2.csv"

df = pd.read_csv(csv_path)

for i, row in df.iterrows():
    tif = row.get('tif_name')
    df.at[i, "tif_name"] = f"{row['prefix_1']}_{row['prefix_2']}_021.tif"

df.to_csv(csv_path, index=False)
print(f"[DONE] tif_name filled → {csv_path}")

# if pd.isna(tif) or str(tif).strip() == "":
