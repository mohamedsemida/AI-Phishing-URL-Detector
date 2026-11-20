
import pandas as pd
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
DATA = BASE / "Dataset" / "All.csv"
OUT = BASE / "Dataset_Audit" / "Duplicate_Leakage_Audit"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
target = "URL_Type_obf_Type"

binary = df[df[target].isin(["benign", "phishing"])].copy()
features = binary.drop(columns=[target])

print("=" * 65)
print("DUPLICATE AND DATA LEAKAGE AUDIT")
print("=" * 65)

# 1. Duplicate counts
print("\n[1] DUPLICATE COUNTS")
print("Duplicate full rows:", binary.duplicated().sum())
print("Duplicate feature rows:", features.duplicated().sum())

# 2. Group identical feature rows
grouped = (
    binary.groupby(list(features.columns), dropna=False)[target]
    .agg(
        records="size",
        unique_labels="nunique",
        labels=lambda x: " | ".join(sorted(x.unique()))
    )
    .reset_index()
)

duplicate_groups = grouped[grouped["records"] > 1].copy()
conflicting_groups = grouped[grouped["unique_labels"] > 1].copy()

print("\n[2] DUPLICATE FEATURE GROUPS")
print("Total groups with repeated features:", len(duplicate_groups))
print("Extra repeated records:", int((duplicate_groups["records"] - 1).sum()))

print("\n[3] CONFLICTING LABEL GROUPS")
print("Groups with identical features but different labels:", len(conflicting_groups))
print("Records in conflicting groups:", int(conflicting_groups["records"].sum()))

if len(conflicting_groups) > 0:
    print("\nConflicting examples:")
    print(conflicting_groups[["records", "unique_labels", "labels"]].head(20).to_string(index=False))

# 4. Same-class vs cross-class duplicate groups
same_class_groups = duplicate_groups[duplicate_groups["unique_labels"] == 1]
cross_class_groups = duplicate_groups[duplicate_groups["unique_labels"] > 1]

print("\n[4] DUPLICATE GROUP TYPES")
print("Same-label duplicate groups:", len(same_class_groups))
print("Cross-label duplicate groups:", len(cross_class_groups))

# 5. Save reports
duplicate_groups.to_csv(OUT / "duplicate_feature_groups.csv", index=False)
conflicting_groups.to_csv(OUT / "conflicting_label_groups.csv", index=False)

summary = pd.DataFrame({
    "metric": [
        "binary_rows",
        "duplicate_full_rows",
        "duplicate_feature_rows",
        "duplicate_feature_groups",
        "extra_repeated_records",
        "conflicting_label_groups",
        "records_in_conflicting_groups",
        "same_label_duplicate_groups",
        "cross_label_duplicate_groups"
    ],
    "value": [
        len(binary),
        int(binary.duplicated().sum()),
        int(features.duplicated().sum()),
        len(duplicate_groups),
        int((duplicate_groups["records"] - 1).sum()),
        len(conflicting_groups),
        int(conflicting_groups["records"].sum()),
        len(same_class_groups),
        len(cross_class_groups)
    ]
})

summary.to_csv(OUT / "duplicate_leakage_summary.csv", index=False)

print("\nReports saved to:", OUT)
print("\nDUPLICATE AND LEAKAGE AUDIT COMPLETED.")
