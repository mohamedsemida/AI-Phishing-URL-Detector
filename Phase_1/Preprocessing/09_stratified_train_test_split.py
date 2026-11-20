
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

INPUT = (
    BASE / "Preprocessing" /
    "Modeling_Preparation" /
    "03_modeling_input_before_split.csv"
)

OUTPUT_DIR = BASE / "Preprocessing" / "Train_Test_Split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "URL_Type_obf_Type"
RANDOM_STATE = 42
TEST_SIZE = 0.20

print("=" * 65)
print("STRATIFIED TRAIN TEST SPLIT")
print("=" * 65)

# 1. Load dataset
df = pd.read_csv(INPUT)

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\n[1] ORIGINAL DATA")
print("Total samples:", len(df))
print("Total features:", X.shape[1])

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# 3. Recombine features and target
train_df = X_train.copy()
train_df[TARGET] = y_train

test_df = X_test.copy()
test_df[TARGET] = y_test

# 4. Save datasets
train_path = OUTPUT_DIR / "train.csv"
test_path = OUTPUT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# 5. Display class distribution
print("\n[2] TRAINING SET")
print("Samples:", len(train_df))
print("Class distribution:")
print(y_train.value_counts())
print("\nClass percentages:")
print((y_train.value_counts(normalize=True) * 100).round(2))

print("\n[3] TESTING SET")
print("Samples:", len(test_df))
print("Class distribution:")
print(y_test.value_counts())
print("\nClass percentages:")
print((y_test.value_counts(normalize=True) * 100).round(2))

# 6. Verify split
print("\n[4] SPLIT VALIDATION")

print("Training + Testing:", len(train_df) + len(test_df))
print("Original samples:", len(df))

print("Training duplicates:", train_df.duplicated().sum())
print("Testing duplicates:", test_df.duplicated().sum())

train_test_overlap = pd.merge(
    train_df,
    test_df,
    how="inner"
)

print("Exact rows shared between train and test:",
      len(train_test_overlap))

print("Training missing values:", train_df.isna().sum().sum())
print("Testing missing values:", test_df.isna().sum().sum())

assert len(train_df) + len(test_df) == len(df)
assert len(train_test_overlap) == 0
assert set(y_train.unique()) == set(y.unique())
assert set(y_test.unique()) == set(y.unique())

print("\n[5] SAVED FILES")
print(train_path)
print(test_path)

print("\nSTRATIFIED SPLIT COMPLETED SUCCESSFULLY.")
