
import numpy as np
import joblib
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

DATA = BASE / "Preprocessing" / "Pipeline_Results"

pipeline = joblib.load(DATA / "preprocessing_pipeline.joblib")

X_train = np.load(DATA / "X_train_processed.npy")
X_test = np.load(DATA / "X_test_processed.npy")

y_train = np.load(DATA / "y_train.npy", allow_pickle=True)
y_test = np.load(DATA / "y_test.npy", allow_pickle=True)

print("=" * 65)
print("PREPROCESSING VALIDATION")
print("=" * 65)

# 1. Shape validation
print("\n[1] SHAPE VALIDATION")
print("Train:", X_train.shape)
print("Test:", X_test.shape)
print("Train labels:", y_train.shape)
print("Test labels:", y_test.shape)

assert X_train.shape[1] == X_test.shape[1]
assert X_train.shape[0] == len(y_train)
assert X_test.shape[0] == len(y_test)

# 2. Invalid values
print("\n[2] INVALID VALUE VALIDATION")
print("Train NaN:", np.isnan(X_train).sum())
print("Test NaN:", np.isnan(X_test).sum())
print("Train Inf:", np.isinf(X_train).sum())
print("Test Inf:", np.isinf(X_test).sum())

assert np.isfinite(X_train).all()
assert np.isfinite(X_test).all()

# 3. Pipeline validation
print("\n[3] PIPELINE VALIDATION")
print("Pipeline steps:", list(pipeline.named_steps.keys()))
print("Imputer strategy:", pipeline.named_steps["imputer"].strategy)
print("Missing indicators:", pipeline.named_steps["imputer"].add_indicator)
print("Scaler:", type(pipeline.named_steps["scaler"]).__name__)

# 4. Training scaler statistics
scaler = pipeline.named_steps["scaler"]

print("\n[4] SCALER STATISTICS")
print("Number of scaler features:", len(scaler.mean_))
print("Training mean statistics available:", np.isfinite(scaler.mean_).all())

# 5. Labels validation
print("\n[5] LABEL VALIDATION")
print("Training classes:", np.unique(y_train))
print("Testing classes:", np.unique(y_test))

assert set(np.unique(y_train)) == {"benign", "phishing"}
assert set(np.unique(y_test)) == {"benign", "phishing"}

print("\nALL VALIDATION CHECKS PASSED.")
