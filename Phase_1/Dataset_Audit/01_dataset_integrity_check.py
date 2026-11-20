import pandas as pd
import numpy as np
from pathlib import Path

project = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
file_path = project / "Dataset" / "All.csv"
output_dir = project / "Dataset_Audit" / "Integrity_Results"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ISCX-URL2016 DATASET INTEGRITY CHECK")
print("=" * 60)

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

df = pd.read_csv(file_path)

print(f"\nFile size: {file_path.stat().st_size / (1024 * 1024):.2f} MB")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")
print(f"Target column exists: {'URL_Type_obf_Type' in df.columns}")

print("\n--- DATASET STRUCTURE ---")
print(df.info())

print("\n--- MISSING VALUES ---")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False))
print(f"Total missing cells: {missing.sum():,}")

print("\n--- DUPLICATES ---")
print(f"Exact duplicate rows: {df.duplicated().sum():,}")

print("\n--- TARGET DISTRIBUTION ---")
if "URL_Type_obf_Type" in df.columns:
    print(df["URL_Type_obf_Type"].value_counts(dropna=False))
    print("\nPercentages:")
    print((df["URL_Type_obf_Type"].value_counts(normalize=True, dropna=False) * 100).round(2))

print("\n--- INFINITE VALUES ---")
numeric = df.select_dtypes(include=np.number)
inf_counts = pd.Series(
    np.isinf(numeric).sum(),
    index=numeric.columns
)
print(inf_counts[inf_counts > 0])

print("\n--- CONSTANT FEATURES ---")
features = df.drop(columns=["URL_Type_obf_Type"], errors="ignore")
constant = [col for col in features.columns if features[col].nunique(dropna=False) <= 1]
print(constant if constant else "No constant features found.")

print("\n--- REPORTS ---")
missing.rename("missing_count").to_csv(output_dir / "missing_values.csv")
df["URL_Type_obf_Type"].value_counts(dropna=False).to_csv(
    output_dir / "class_distribution.csv"
) if "URL_Type_obf_Type" in df.columns else None

summary = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "missing_cells",
        "exact_duplicate_rows",
        "infinite_cells",
        "constant_features"
    ],
    "value": [
        len(df),
        len(df.columns),
        int(df.isnull().sum().sum()),
        int(df.duplicated().sum()),
        int(np.isinf(numeric).sum().sum()),
        len(constant)
    ]
})
summary.to_csv(output_dir / "integrity_summary.csv", index=False)

print(f"Reports saved to: {output_dir}")
print("\nINTEGRITY CHECK COMPLETED.")
