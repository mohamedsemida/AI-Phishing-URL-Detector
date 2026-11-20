
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
DATA = BASE / "Preprocessing" / "01_binary_deduplicated.csv"
OUT = BASE / "Preprocessing" / "Feature_Quality_Audit"

OUT.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"
df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])

print("=" * 65)
print("FEATURE QUALITY AUDIT")
print("=" * 65)

# 1. Constant features
constant = [col for col in X.columns if X[col].nunique(dropna=False) <= 1]

print("\n[1] CONSTANT FEATURES")
print(constant if constant else "None")

# 2. Near-constant features (at least 99.9% same value)
near_constant = []

for col in X.columns:
    proportions = X[col].value_counts(dropna=False, normalize=True)
    if len(proportions) > 0 and proportions.iloc[0] >= 0.999:
        near_constant.append({
            "feature": col,
            "dominant_value": str(proportions.index[0]),
            "dominant_percentage": round(proportions.iloc[0] * 100, 4)
        })

near_df = pd.DataFrame(near_constant)

print("\n[2] NEAR-CONSTANT FEATURES (>=99.9%)")
if len(near_df):
    print(near_df.to_string(index=False))
else:
    print("None")

# 3. Exact duplicate columns
duplicate_columns = []

columns = list(X.columns)

for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        if X[columns[i]].equals(X[columns[j]]):
            duplicate_columns.append({
                "feature_1": columns[i],
                "feature_2": columns[j]
            })

duplicate_df = pd.DataFrame(duplicate_columns)

print("\n[3] EXACT DUPLICATE FEATURES")
if len(duplicate_df):
    print(duplicate_df.to_string(index=False))
else:
    print("None")

# 4. Missing and infinite values
numeric = X.select_dtypes(include=np.number)
numeric_clean = numeric.replace([np.inf, -np.inf], np.nan)

quality = pd.DataFrame({
    "missing_count": X.isna().sum(),
    "unique_values": X.nunique(dropna=False)
})

quality["infinite_count"] = 0
for col in numeric.columns:
    quality.loc[col, "infinite_count"] = int(np.isinf(numeric[col]).sum())

print("\n[4] FEATURES WITH MISSING OR INFINITE VALUES")
print(quality[
    (quality["missing_count"] > 0) |
    (quality["infinite_count"] > 0)
].sort_values(
    ["missing_count", "infinite_count"],
    ascending=False
).to_string())

# 5. High correlation
corr = numeric_clean.corr().abs()

upper = corr.where(
    np.triu(np.ones(corr.shape), k=1).astype(bool)
)

high_corr = (
    upper.stack()
    .reset_index()
)

high_corr.columns = ["feature_1", "feature_2", "correlation"]
high_corr = high_corr[high_corr["correlation"] >= 0.95]
high_corr = high_corr.sort_values("correlation", ascending=False)

print("\n[5] HIGHLY CORRELATED FEATURES (>= 0.95)")
if len(high_corr):
    print(high_corr.to_string(index=False))
else:
    print("None")

# 6. Save reports
quality.to_csv(OUT / "feature_quality_summary.csv")
near_df.to_csv(OUT / "near_constant_features.csv", index=False)
duplicate_df.to_csv(OUT / "exact_duplicate_features.csv", index=False)
high_corr.to_csv(OUT / "high_correlation_features.csv", index=False)

pd.DataFrame({"constant_feature": constant}).to_csv(
    OUT / "constant_features.csv", index=False
)

print("\nReports saved to:", OUT)
print("\nFEATURE QUALITY AUDIT COMPLETED.")
