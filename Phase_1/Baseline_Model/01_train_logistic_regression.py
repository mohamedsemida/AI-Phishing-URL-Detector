
import numpy as np
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"

DATA = BASE / "Preprocessing" / "Pipeline_Results"
OUTPUT = BASE / "Baseline_Model" / "Results"

OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("BASELINE MODEL - LOGISTIC REGRESSION")
print("=" * 65)

# 1. Load processed data
X_train = np.load(DATA / "X_train_processed.npy")
X_test = np.load(DATA / "X_test_processed.npy")

y_train = np.load(DATA / "y_train.npy", allow_pickle=True)
y_test = np.load(DATA / "y_test.npy", allow_pickle=True)

print("\n[1] DATA LOADED")
print("Training:", X_train.shape)
print("Testing:", X_test.shape)

# 2. Initialize model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# 3. Train model
print("\n[2] TRAINING MODEL")
model.fit(X_train, y_train)

print("Training completed successfully.")

# 4. Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

phishing_index = list(model.classes_).index("phishing")
phishing_prob = y_prob[:, phishing_index]

# 5. Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    pos_label="phishing",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    pos_label="phishing",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    pos_label="phishing",
    zero_division=0
)

roc_auc = roc_auc_score(
    (y_test == "phishing").astype(int),
    phishing_prob
)

pr_auc = average_precision_score(
    (y_test == "phishing").astype(int),
    phishing_prob
)

# 6. Display results
print("\n[3] TESTING RESULTS")
print("-" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

print("\n[4] CONFUSION MATRIX")
print("Labels order: benign, phishing")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["benign", "phishing"]
)

print(cm)

print("\n[5] CLASSIFICATION REPORT")
report = classification_report(
    y_test,
    y_pred,
    labels=["benign", "phishing"],
    zero_division=0
)

print(report)

# 7. Save model
model_path = OUTPUT / "logistic_regression_baseline.joblib"
joblib.dump(model, model_path)

# 8. Save evaluation report
report_path = OUTPUT / "baseline_evaluation.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("LOGISTIC REGRESSION - BASELINE EVALUATION\n")
    file.write("=" * 50 + "\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-Score: {f1:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n")
    file.write(f"PR-AUC: {pr_auc:.4f}\n\n")
    file.write("Confusion Matrix:\n")
    file.write(str(cm))
    file.write("\n\nClassification Report:\n")
    file.write(report)

print("\n[6] SAVED FILES")
print("Model:", model_path)
print("Evaluation:", report_path)

print("\nBASELINE MODEL COMPLETED.")
