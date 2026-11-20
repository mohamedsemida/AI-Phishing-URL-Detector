
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

DATA = BASE / "Preprocessing" / "Reduced_Features" / "02_binary_reduced_features.csv"
OUT = BASE / "Preprocessing" / "Missing_Sentinel_Overlap"

OUT.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"

df = pd.read_csv(DATA)
X = df.drop(columns=[TARGET])

rows = []

print("=" * 65)
print("MISSING AND SENTINEL OVERLAP AUDIT")
print("=" * 65)

for feature in X.columns:

    series = X[feature]

    missing_mask = series.isna()

    if pd.api.types.is_numeric_dtype(series):
        infinite_mask = pd.Series(
            np.isinf(series.to_numpy(dtype=float)),
            index=series.index
        )
    else:
        infinite_mask = pd.Series(False, index=series.index)

    minus_one_mask = series.eq(-1).fillna(False)

    missing_count = int(missing_mask.sum())
    infinite_count = int(infinite_mask.sum())
    minus_one_count = int(minus_one_mask.sum())

    missing_and_minus_one = int(
        (missing_mask & minus_one_mask).sum()
    )

    missing_or_minus_one = int(
        (missing_mask | minus_one_mask).sum()
    )

    invalid_mask = missing_mask | infinite_mask | minus_one_mask

    rows.append({
        "feature": feature,
        "missing_count": missing_count,
        "infinite_count": infinite_count,
        "minus_one_count": minus_one_count,
        "missing_and_minus_one": missing_and_minus_one,
        "missing_or_minus_one": missing_or_minus_one,
        "rows_with_any_issue": int(invalid_mask.sum()),
        "valid_rows": int((~invalid_mask).sum()),
        "total_rows": len(df)
    })

report = pd.DataFrame(rows)

report = report[
    (report["missing_count"] > 0) |
    (report["infinite_count"] > 0) |
    (report["minus_one_count"] > 0)
].sort_values(
    "rows_with_any_issue",
    ascending=False
)

print("\n[1] FEATURE-LEVEL OVERLAP")
print(report.to_string(index=False))

print("\n[2] TOTAL AFFECTED ROWS PER FEATURE")
print(
    report[
        ["feature", "rows_with_any_issue", "valid_rows", "total_rows"]
    ].to_string(index=False)
)

# Class-specific invalid-value counts
class_rows = []

for feature in X.columns:
    for label in ["benign", "phishing"]:

        subset = df[df[TARGET] == label]
        values = subset[feature]

        missing_count = int(values.isna().sum())

        if pd.api.types.is_numeric_dtype(values):
            infinite_count = int(
                np.isinf(values.to_numpy(dtype=float)).sum()
            )
        else:
            infinite_count = 0

        minus_one_count = int(
            values.eq(-1).fillna(False).sum()
        )

        if missing_count or infinite_count or minus_one_count:
            class_rows.append({
                "feature": feature,
                "class": label,
                "missing_count": missing_count,
                "infinite_count": infinite_count,
                "minus_one_count": minus_one_count
            })

class_report = pd.DataFrame(class_rows)

report.to_csv(
    OUT / "missing_sentinel_overlap.csv",
    index=False
)

class_report.to_csv(
    OUT / "missing_sentinel_by_class.csv",
    index=False
)

print("\nReports saved to:", OUT)
print("\nOVERLAP AUDIT COMPLETED.")
