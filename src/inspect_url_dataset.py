
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR / "data" / "raw"
    / "phishing+websites" / "url_data"
    / "phishing_url_dataset.csv"
)

print("=" * 55)
print("PHISHGUARD AI - URL DATASET INSPECTION")
print("=" * 55)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"CSV file not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five records:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes.to_string())

print("\nMissing values:")
print(df.isna().sum().to_string())

for column in df.columns:
    if df[column].nunique(dropna=False) <= 10:
        print(f"\nPossible label column: {column}")
        print(df[column].value_counts(dropna=False).to_string())

print("\nURL DATASET INSPECTION COMPLETE")
