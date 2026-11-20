
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
DATA = BASE / "Dataset" / "All.csv"
OUT = BASE / "Dataset_Audit" / "Binary_Scope_Audit"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
target = "URL_Type_obf_Type"

# Select only the two classes required by the project
binary = df[df[target].isin(["benign", "phishing"])].copy()
X = binary.drop(columns=[target])

print("=" * 65)
print("ISCX-URL2016 BINARY SCOPE AUDIT")
print("=" * 65)

# 1. Dataset scope
print("\n[1] BINARY DATASET SIZE")
print("Original rows:", len(df))
print("Selected rows:", len(binary))
print("Features:", X.shape[1])

# 2. Class distribution
print("\n[2] CLASS DISTRIBUTION")
counts = binary[target].value_counts()
print(counts)
print("\nPercentages:")
print((binary[target].value_counts(normalize=True) * 100).round(2))

# 3. Exact duplicate rows
print("\n[3] EXACT DUPLICATE ROWS")
print("All columns:", binary.duplicated().sum())
print("Features only:", X.duplicated().sum())

# 4. Missing values
print("\n[4] MISSING VALUES")
missing = X.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))
print("Total missing cells:", int(missing.sum()))

# 5. Infinite values
print("\n[5] INFINITE VALUES")
numeric = X.select_dtypes(include=np.number)
infinite = pd.Series(
    np.isinf(numeric).sum(),
    index=numeric.columns
)
print(infinite[infinite > 0])
print("Total infinite cells:", int(infinite.sum()))

# 6. Negative one values
print("\n[6] -1 VALUES BY FEATURE")
minus_one = (X == -1).sum()
minus_one = minus_one[minus_one > 0].sort_values(ascending=False)
print(minus_one)

# 7. -1 distribution by class for affected features
print("\n[7] -1 DISTRIBUTION BY CLASS")
rows = []

for feature in minus_one.index:
    for label in ["benign", "phishing"]:
        class_data = binary[binary[target] == label]
        count = int((class_data[feature] == -1).sum())
        rows.append({
            "feature": feature,
            "class": label,
            "minus_one_count": count,
            "class_records": len(class_data),
            "percentage": round(count / len(class_data) * 100, 3)
        })

minus_one_by_class = pd.DataFrame(rows)
print(minus_one_by_class.to_string(index=False))

# 8. Save reports
counts.rename("records").to_csv(OUT / "binary_class_distribution.csv")
missing.rename("missing_count").to_csv(OUT / "binary_missing_values.csv")
infinite.rename("infinite_count").to_csv(OUT / "binary_infinite_values.csv")
minus_one_by_class.to_csv(
    OUT / "minus_one_by_class.csv", index=False
)

summary = pd.DataFrame({
    "metric": [
        "original_rows",
        "binary_rows",
        "feature_count",
        "duplicate_full_rows",
        "duplicate_feature_rows",
        "missing_cells",
        "infinite_cells"
    ],
    "value": [
        len(df),
        len(binary),
        X.shape[1],
        int(binary.duplicated().sum()),
        int(X.duplicated().sum()),
        int(X.isna().sum().sum()),
        int(np.isinf(numeric).sum().sum())
    ]
})

summary.to_csv(OUT / "binary_scope_summary.csv", index=False)

print("\nReports saved to:", OUT)
print("\nBINARY SCOPE AUDIT COMPLETED.")
