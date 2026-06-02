import pandas as pd

CSV_PATH = "./metadata_JPSS-2.csv"  # 필요하면 경로 수정

cols_to_clear = [
    "KR_Sea_overlap",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
]

df = pd.read_csv(CSV_PATH)

# 컬럼 존재 여부 체크 (없으면 에러 나게)
missing = [c for c in cols_to_clear if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# 값 전부 삭제 (NaN으로)
df[cols_to_clear] = pd.NA

# 다시 저장
df.to_csv(CSV_PATH, index=False)

print("[DONE] specified columns cleared.")