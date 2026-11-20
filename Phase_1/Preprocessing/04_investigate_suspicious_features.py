
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path.home() / "Desktop" / "AI_Phishing_URL_Detector" / "Phase_1"
DATA = BASE / "Preprocessing" / "01_binary_deduplicated.csv"
OUT = BASE / "Preprocessing" / "Feature_Investigation"

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

print("=" * 65)
print("SUSPICIOUS FEATURE INVESTIGATION")
print("=" * 65)

# 1. Inspect the two rare records
print("\n[1] RARE FEATURE RECORDS")

mask = (df["executable"] == 1) | (df["isPortEighty"] == 0)

rare_records = df.loc[mask].copy()

print("Number of records:", len(rare_records))

columns_to_show = [
    "URL_Type_obf_Type",
    "executable",
    "isPortEighty",
    "domain_token_count",
    "tld",
    "SymbolCount_Domain",
    "pathLength",
    "subDirLen",
    "urlLen"
]

print(rare_records[columns_to_show].to_string(index=True))

rare_records.to_csv(
    OUT / "rare_feature_records.csv",
    index=True
)

# 2. Investigate perfect correlation
print("\n[2] DOMAIN FEATURE RELATIONSHIP")

x = pd.to_numeric(df["domain_token_count"], errors="coerce")
y = pd.to_numeric(df["SymbolCount_Domain"], errors="coerce")

valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)

x_valid = x[valid]
y_valid = y[valid]

print("Correlation:", x_valid.corr(y_valid))
print("Unique domain_token_count values:", x_valid.nunique())
print("Unique SymbolCount_Domain values:", y_valid.nunique())
print("Unique value pairs:", len(pd.DataFrame({
    "x": x_valid,
    "y": y_valid
}).drop_duplicates()))

# Fit a linear relationship for investigation only
slope, intercept = np.polyfit(x_valid, y_valid, 1)
predicted = slope * x_valid + intercept
residuals = y_valid - predicted

print("Linear slope:", slope)
print("Linear intercept:", intercept)
print("Maximum absolute residual:", abs(residuals).max())

print("\nSample value pairs:")
print(pd.DataFrame({
    "domain_token_count": x_valid,
    "SymbolCount_Domain": y_valid
}).drop_duplicates().head(15).to_string(index=False))

# Save relationship data
pd.DataFrame({
    "domain_token_count": x_valid,
    "SymbolCount_Domain": y_valid
}).to_csv(
    OUT / "domain_feature_relationship.csv",
    index=False
)

print("\nReports saved to:", OUT)
print("\nINVESTIGATION COMPLETED.")
