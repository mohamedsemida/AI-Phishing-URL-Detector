
import pandas as pd
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

SOURCE = BASE / "Preprocessing" / "01_binary_deduplicated.csv"
OUT_DIR = BASE / "Preprocessing" / "Reduced_Features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"

DROP_FEATURES = [
    "ISIpAddressInDomainName",
    "tld",
    "SymbolCount_Domain",
    "subDirLen"
]

df = pd.read_csv(SOURCE)

print("=" * 65)
print("REDUCED FEATURE DATASET CREATION")
print("=" * 65)

print("Original columns:", df.shape[1])
print("Original features:", df.shape[1] - 1)

reduced = df.drop(columns=DROP_FEATURES)

print("\nRemoved features:")
for feature in DROP_FEATURES:
    print("-", feature)

print("\nRemaining columns:", reduced.shape[1])
print("Remaining features:", reduced.shape[1] - 1)

print("\nClass distribution:")
print(reduced[TARGET].value_counts())

output_file = OUT_DIR / "02_binary_reduced_features.csv"
reduced.to_csv(output_file, index=False)

print("\nSaved to:", output_file)
print("\nREDUCED FEATURE DATASET CREATED.")
