
import pandas as pd
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

SOURCE = BASE / "Dataset" / "All.csv"
OUTPUT_DIR = BASE / "Preprocessing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"

df = pd.read_csv(SOURCE)

# Keep only Benign and Phishing
binary = df[df[TARGET].isin(["benign", "phishing"])].copy()

before = len(binary)

# Remove exact duplicate rows
deduplicated = binary.drop_duplicates().copy()

after = len(deduplicated)
removed = before - after

# Save the intermediate dataset
output_file = OUTPUT_DIR / "01_binary_deduplicated.csv"
deduplicated.to_csv(output_file, index=False)

print("=" * 60)
print("BINARY DATASET DEDUPLICATION")
print("=" * 60)

print("Original dataset rows:", len(df))
print("Binary dataset before deduplication:", before)
print("Rows after deduplication:", after)
print("Duplicate rows removed:", removed)

print("\nClass distribution after deduplication:")
print(deduplicated[TARGET].value_counts())

print("\nRemaining duplicate rows:", deduplicated.duplicated().sum())

print("\nSaved to:", output_file)
print("\nDEDUPLICATION COMPLETED.")
