
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

DATA = BASE / "Preprocessing" / "Reduced_Features" / "02_binary_reduced_features.csv"
OUT = BASE / "Preprocessing" / "Missing_Sentinel_Audit"

OUT.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])

print("=" * 65)
print("MISSING, SENTINEL AND INFINITE AUDIT")
print("=" * 65)

# 1. Missing values
print("\n[1] MISSING VALUES")

missing = X.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))
print("Total missing cells:", int(missing.sum()))

# 2. Infinite values
print("\n[2] INFINITE VALUES")

numeric = X.select_dtypes(include=np.number)
infinite = pd.Series(
    np.isinf(numeric).sum(),
    index=numeric.columns
)

print(infinite[infinite > 0])
print("Total infinite cells:", int(infinite.sum()))

# 3. -1 values
print("\n[3] -1 VALUES")

minus_one = (X == -1).sum()
minus_one = minus_one[minus_one > 0].sort_values(ascending=False)

print(minus_one)
print("Total -1 cells:", int(minus_one.sum()))

# 4. -1 distribution by class
print("\n[4] -1 DISTRIBUTION BY CLASS")

rows = []

for feature in minus_one.index:
    for label in ["benign", "phishing"]:
        subset = df[df[TARGET] == label]
        count = int((subset[feature] == -1).sum())

        rows.append({
            "feature": feature,
            "class": label,
            "minus_one_count": count,
            "class_records": len(subset),
            "percentage": round(count / len(subset) * 100, 3)
        })

minus_by_class = pd.DataFrame(rows)
print(minus_by_class.to_string(index=False))

# 5. Missing values by class
print("\n[5] MISSING VALUES BY CLASS")

missing_rows = []

for feature in X.columns:
    for label in ["benign", "phishing"]:
        subset = df[df[TARGET] == label]
        count = int(subset[feature].isna().sum())

        if count > 0:
            missing_rows.append({
                "feature": feature,
                "class": label,
                "missing_count": count,
                "class_records": len(subset),
                "percentage": round(count / len(subset) * 100, 3)
            })

missing_by_class = pd.DataFrame(missing_rows)

if len(missing_by_class):
    print(missing_by_class.to_string(index=False))
else:
    print("No missing values.")

# 6. Save reports
missing.rename("missing_count").to_csv(
    OUT / "missing_values.csv"
)

infinite.rename("infinite_count").to_csv(
    OUT / "infinite_values.csv"
)

minus_one.rename("minus_one_count").to_csv(
    OUT / "minus_one_values.csv"
)

minus_by_class.to_csv(
    OUT / "minus_one_by_class.csv",
    index=False
)

missing_by_class.to_csv(
    OUT / "missing_by_class.csv",
    index=False
)

print("\nReports saved to:", OUT)
print("\nMISSING AND SENTINEL AUDIT COMPLETED.")
