import pandas as pd

# =========================
# 설정
# =========================
SNPP_path = "./metadata_S-NPP.csv"
JPSS1_path = "./metadata_JPSS-1.csv"
JPSS2_path = "./metadata_JPSS-2.csv"

cols = ["prefix_1", "scan_start_HHMM"]
# cols = ["scan_start_HHMMSS"]

# =========================
# CSV 로드
# =========================
df1 = pd.read_csv(SNPP_path, usecols=cols)
df2 = pd.read_csv(JPSS1_path, usecols=cols)
df3 = pd.read_csv(JPSS2_path, usecols=cols)

# (선택) 문자열 정규화가 필요하면
# for df in (df1, df2, df3):
#     for c in cols:
#         df[c] = df[c].astype(str).str.strip()

# =========================
# 2개 변수 → tuple key
# =========================
set1 = set(map(tuple, df1[cols].dropna().values))
set2 = set(map(tuple, df2[cols].dropna().values))
set3 = set(map(tuple, df3[cols].dropna().values))

# =========================
# 경우별 분류
# =========================
only_1 = set1 - set2 - set3
only_2 = set2 - set1 - set3
only_3 = set3 - set1 - set2

only_12 = (set1 & set2) - set3
only_13 = (set1 & set3) - set2
only_23 = (set2 & set3) - set1

all_123 = set1 & set2 & set3

# =========================
# 통계 출력
# =========================
stats = {
    "S-NPP only": len(only_1),
    "JPSS-1 only": len(only_2),
    "JPSS-2 only": len(only_3),
    "S-NPP & JPSS-1 only": len(only_12),
    "S-NPP & JPSS-2 only": len(only_13),
    "JPSS-1 & JPSS-2 only": len(only_23),
    "S-NPP & JPSS-1 & JPSS-2": len(all_123),
}
cols[0] = "DATE & scan start HH:MM"
print(f"\n=== Statistics ({cols[0]}) ===")
for k, v in stats.items():
    print(f"{k:25s}: {v}")

# =========================
# (선택) 결과를 CSV로 저장
# =========================
pd.DataFrame(list(only_1), columns=cols).to_csv(cols[0] + "_only_S-NPP.csv", index=False)
pd.DataFrame(list(only_12), columns=cols).to_csv(cols[0] + "_only_S-NPP_JPSS-1.csv", index=False)
pd.DataFrame(list(all_123), columns=cols).to_csv(cols[0] + "_common_all.csv", index=False)