
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

INPUT_DIR = BASE / "Preprocessing" / "Train_Test_Split"
OUTPUT_DIR = BASE / "Preprocessing" / "Pipeline_Results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = INPUT_DIR / "train.csv"
TEST_FILE = INPUT_DIR / "test.csv"

TARGET = "URL_Type_obf_Type"

print("=" * 65)
print("PREPROCESSING PIPELINE")
print("=" * 65)

# 1. Load split data
train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

X_train = train.drop(columns=[TARGET])
X_test = test.drop(columns=[TARGET])

y_train = train[TARGET]
y_test = test[TARGET]

print("\n[1] INPUT DATA")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# 2. Create pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="median",
        add_indicator=True
    )),
    ("scaler", StandardScaler())
])

# 3. Fit ONLY on training data
X_train_processed = pipeline.fit_transform(X_train)

# 4. Transform testing data without fitting
X_test_processed = pipeline.transform(X_test)

# 5. Validate results
print("\n[2] PROCESSED DATA")
print("Training shape:", X_train_processed.shape)
print("Testing shape:", X_test_processed.shape)

print("\n[3] MISSING VALUE CHECK")
print("Training NaN:", np.isnan(X_train_processed).sum())
print("Testing NaN:", np.isnan(X_test_processed).sum())

print("\n[4] INFINITY CHECK")
print("Training Inf:", np.isinf(X_train_processed).sum())
print("Testing Inf:", np.isinf(X_test_processed).sum())

# 6. Save processed arrays and labels
np.save(OUTPUT_DIR / "X_train_processed.npy", X_train_processed)
np.save(OUTPUT_DIR / "X_test_processed.npy", X_test_processed)

np.save(OUTPUT_DIR / "y_train.npy", y_train.to_numpy())
np.save(OUTPUT_DIR / "y_test.npy", y_test.to_numpy())

# 7. Save fitted pipeline
joblib.dump(pipeline, OUTPUT_DIR / "preprocessing_pipeline.joblib")

print("\n[5] SAVED FILES")
for file in OUTPUT_DIR.iterdir():
    print(file.name)

print("\nPREPROCESSING PIPELINE COMPLETED.")
