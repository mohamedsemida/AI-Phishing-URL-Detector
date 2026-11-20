# AI-Based Phishing URL Detection System

An AI-based cybersecurity project that uses machine learning and lexical URL features to distinguish between benign and phishing URLs.

## 1. Project Overview

Phishing attacks use malicious URLs to deceive users and steal sensitive information, including banking credentials and personal data.

This project explores the use of machine learning to identify phishing-related URL patterns through lexical feature analysis.

The project is developed in multiple phases. This repository currently documents **Phase 1: Dataset Analysis, Preprocessing, and Baseline Model Development**.

## 2. Project Objectives

- Analyze and evaluate available phishing URL datasets.
- Select a suitable dataset for binary classification.
- Perform data cleaning and preprocessing.
- Investigate missing values, duplicate records, and feature quality.
- Build a baseline machine learning model.
- Evaluate model performance using classification metrics and cross-validation.

## 3. Dataset

**Selected Dataset:** ISCX-URL2016

The project uses the Benign and Phishing classes from the original dataset.

| Description | Value |
|---|---:|
| Original dataset records | 36,707 |
| Original features | 79 |
| Selected classes | Benign, Phishing |
| Binary dataset before deduplication | 15,367 |
| Dataset after deduplication | 14,823 |
| Final features | 73 |

The final binary dataset contains:
- Benign: 7,464 records
- Phishing: 7,359 records

**Important:** The repository does not include the original dataset file. Obtain it from the official ISCX-URL2016 dataset source and provide `All.csv` when running the notebook.

## 4. Data Preprocessing

The preprocessing workflow includes:

1. Selecting Benign and Phishing records.
2. Removing duplicate records.
3. Auditing feature quality and redundancy.
4. Removing constant, near-constant, and redundant features.
5. Investigating missing and infinite values.
6. Replacing infinite values with missing values.
7. Preserving undocumented `-1` values rather than assuming they represent missing data.
8. Splitting the dataset into training and testing sets using stratified sampling.
9. Applying median imputation and standardization through a preprocessing pipeline.

### Dataset Split

- Training: 80% (11,858 records)
- Testing: 20% (2,965 records)
- Random state: 42
- Stratification: Enabled

## 5. Baseline Machine Learning Model

**Algorithm:** Logistic Regression

The baseline model was trained using a pipeline containing:
- Median Imputation
- Standard Scaling
- Logistic Regression

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 96.29% |
| Precision | 96.20% |
| Recall | 96.33% |
| F1-Score | 96.27% |
| ROC-AUC | 99.35% |
| PR-AUC | 99.40% |

### Confusion Matrix

| Actual / Predicted | Benign | Phishing |
|---|---:|---:|
| Benign | 1,437 | 56 |
| Phishing | 54 | 1,418 |

## 6. Cross-Validation

A 5-fold Stratified Cross-Validation evaluation was performed.

| Metric | Mean Score |
|---|---:|
| Accuracy | 95.96% |
| Precision | 96.40% |
| Recall | 95.43% |
| F1-Score | 95.91% |
| ROC-AUC | 99.12% |
| PR-AUC | 99.15% |

Cross-validation results were also compared with training scores to investigate potential overfitting.

## 7. Repository Structure

```text
AI-Phishing-URL-Detector/
│
├── README.md
├── .gitignore
│
└── Phase_1/
    ├── Baseline_Model/
    │   ├── 01_train_logistic_regression.py
    │   └── Results/
    │
    ├── Dataset_Audit/
    │   ├── Dataset integrity scripts
    │   └── Audit results
    │
    ├── Documentation/
    │   └── AI_Phishing_URL_Detector_Phase_1_Final.ipynb
    │
    ├── Preprocessing/
    │   ├── Preprocessing scripts
    │   └── Audit results
    │
    └── Reports/
        └── Report.docx
```

## 8. How to Run

The main experimental workflow is provided in the Jupyter Notebook.

1. Download the ISCX-URL2016 dataset from its official source.
2. Open the notebook:
   `Phase_1/Documentation/AI_Phishing_URL_Detector_Phase_1_Final.ipynb`
3. Open it using Google Colab.
4. Upload the original `All.csv` dataset when requested.
5. Run the notebook cells in order.
6. Review the preprocessing, training, and evaluation outputs.

## 9. Current Project Scope and Limitations

Phase 1 is an experimental machine learning baseline.

The current implementation classifies records containing pre-extracted lexical URL features. It does not yet accept an arbitrary raw URL and automatically extract its features for real-time prediction.

The reported results are experimental results on the selected dataset and do not represent guaranteed performance against previously unseen real-world phishing campaigns.

Further development will focus on comparing additional machine learning algorithms, model optimization, explainability, and prototype development.

## 10. References

- Mamun, M. S. I., Rathore, M. A., Lashkari, A. H., Stakhanova, N., & Ghorbani, A. A. (2016). *Detecting Malicious URLs Using Lexical Analysis*. Network and System Security (NSS 2016), Lecture Notes in Computer Science, 9955, 467–482.
- DOI: https://doi.org/10.1007/978-3-319-46298-1_30
- Official dataset source: https://www.unb.ca/cic/datasets/url-2016.html

---

**Project Status:** Phase 1 – Baseline Model and Evaluation.
