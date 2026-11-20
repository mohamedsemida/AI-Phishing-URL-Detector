
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

INPUT = BASE / "Preprocessing" / "Reduced_Features" / "02_binary_reduced_features.csv"

OUTPUT_DIR = BASE / "Preprocessing" / "Modeling_Preparation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "03_modeling_input_before_split.csv"

TARGET = "URL_Type_obf_Type"

DROP_FEATURES = [
    "isPortEighty",
    "executable"
]

print("=" * 65)
print("MODELING DATASET PREPARATION")
print("=" * 65)

# 1. Load data
df = pd.read_csv(INPUT)

print("\n[1] ORIGINAL DATA")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Validate target
if TARGET not in df.columns:
    raise ValueError("Target column not found!")

if set(df[TARGET].unique()) != {"benign", "phishing"}:
    raise ValueError("Unexpected target labels!")

# 3. Drop near-constant features
existing_drop = [
    col for col in DROP_FEATURES
    if col in df.columns
]

df = df.drop(columns=existing_drop)

print("\n[2] REMOVED FEATURES")
for col in existing_drop:
    print("-", col)

# 4. Separate features and target
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()

# 5. Convert infinite values to NaN
infinite_before = np.isinf(
    X.select_dtypes(include=np.number)
).sum().sum()

X = X.replace([np.inf, -np.inf], np.nan)

print("\n[3] INFINITE VALUE HANDLING")
print("Infinite values converted to NaN:", int(infinite_before))

# 6. Preserve -1 values
minus_one_count = int(X.eq(-1).sum().sum())

print("\n[4] SENTINEL HANDLING")
print("-1 values preserved:", minus_one_count)

# 7. Check missing values
missing_count = int(X.isna().sum().sum())

print("\n[5] MISSING VALUES")
print("Missing cells remaining:", missing_count)

# 8. Recombine without fitting any preprocessing
prepared = X.copy()
prepared[TARGET] = y

# 9. Save
prepared.to_csv(OUTPUT, index=False)

print("\n[6] FINAL DATASET")
print("Rows:", len(prepared))
print("Features:", len(X.columns))
print("Columns including target:", len(prepared.columns))

print("\nClass distribution:")
print(prepared[TARGET].value_counts())

print("\nRemaining duplicate rows:", prepared.duplicated().sum())

print("\nSaved to:")
print(OUTPUT)

print("\nMODELING DATASET PREPARATION COMPLETED.")
