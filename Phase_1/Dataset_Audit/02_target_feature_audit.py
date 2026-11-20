
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
DATA = BASE / "Dataset" / "All.csv"
OUT = BASE / "Dataset_Audit" / "Target_Feature_Audit"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
target = "URL_Type_obf_Type"

print("=" * 65)
print("TARGET AND FEATURE AUDIT")
print("=" * 65)

# 1. Target validation
print("\n[1] TARGET INFORMATION")
print("Target:", target)
print("Unique labels:", df[target].nunique())
print(df[target].value_counts())

# 2. Feature inventory
print("\n[2] FEATURE INVENTORY")
features = df.drop(columns=[target])

inventory = pd.DataFrame({
    "feature": features.columns,
    "dtype": features.dtypes.astype(str).values,
    "unique_values": features.nunique(dropna=True).values,
    "missing_values": features.isna().sum().values,
    "negative_one_count": [
        int((features[c] == -1).sum())
        if pd.api.types.is_numeric_dtype(features[c]) else 0
        for c in features.columns
    ]
})

print(inventory.to_string(index=False))
inventory.to_csv(OUT / "feature_inventory.csv", index=False)

# 3. Constant features
constant = inventory[inventory["unique_values"] <= 1]
print("\n[3] CONSTANT FEATURES")
print(constant.to_string(index=False))

# 4. Duplicate feature columns
print("\n[4] IDENTICAL FEATURE COLUMNS")
duplicate_columns = []
cols = list(features.columns)

for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if features[cols[i]].equals(features[cols[j]]):
            duplicate_columns.append((cols[i], cols[j]))

for pair in duplicate_columns:
    print(pair)

if not duplicate_columns:
    print("No identical columns found.")

# 5. Negative one summary
print("\n[5] FEATURES CONTAINING -1")
minus_one = inventory[inventory["negative_one_count"] > 0]
print(minus_one[["feature", "negative_one_count"]].to_string(index=False))

# 6. Save audit summary
summary = pd.DataFrame({
    "metric": [
        "target_unique_labels",
        "total_features",
        "constant_features",
        "identical_feature_pairs",
        "features_containing_minus_one"
    ],
    "value": [
        df[target].nunique(),
        len(features.columns),
        len(constant),
        len(duplicate_columns),
        len(minus_one)
    ]
})

summary.to_csv(OUT / "audit_summary.csv", index=False)

print("\nReports saved to:", OUT)
print("TARGET AND FEATURE AUDIT COMPLETED.")
