
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
DATA = BASE / "Preprocessing" / "01_binary_deduplicated.csv"
OUT = BASE / "Preprocessing" / "Feature_Relationship_Review"

OUT.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])

print("=" * 65)
print("FEATURE RELATIONSHIP REVIEW")
print("=" * 65)

# 1. Near-constant feature distributions by class
print("\n[1] NEAR-CONSTANT FEATURES BY CLASS")

near_features = ["executable", "isPortEighty"]

near_rows = []

for feature in near_features:
    print(f"\nFeature: {feature}")

    print(pd.crosstab(
        df[TARGET],
        df[feature],
        dropna=False
    ))

    for label in ["benign", "phishing"]:
        subset = df[df[TARGET] == label]
        counts = subset[feature].value_counts(dropna=False)

        for value, count in counts.items():
            near_rows.append({
                "feature": feature,
                "class": label,
                "value": value,
                "count": count,
                "percentage": round(count / len(subset) * 100, 4)
            })

# 2. Verify exact equality of correlated features
print("\n[2] VERIFY PERFECTLY CORRELATED FEATURES")

pairs = [
    ("domain_token_count", "tld"),
    ("domain_token_count", "SymbolCount_Domain"),
    ("tld", "SymbolCount_Domain"),
    ("pathLength", "subDirLen")
]

pair_rows = []

for col1, col2 in pairs:
    equal = X[col1].equals(X[col2])
    differences = int((X[col1] != X[col2]).sum())

    print(f"\n{col1} vs {col2}")
    print("Exactly equal:", equal)
    print("Different records:", differences)

    pair_rows.append({
        "feature_1": col1,
        "feature_2": col2,
        "exactly_equal": equal,
        "different_records": differences
    })

# 3. Save reports
pd.DataFrame(near_rows).to_csv(
    OUT / "near_constant_class_distribution.csv",
    index=False
)

pd.DataFrame(pair_rows).to_csv(
    OUT / "perfect_correlation_equality_check.csv",
    index=False
)

print("\nReports saved to:", OUT)
print("\nFEATURE RELATIONSHIP REVIEW COMPLETED.")
