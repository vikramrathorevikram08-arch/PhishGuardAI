
from pathlib import Path
import arff
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "phishing+websites" / "Training Dataset.arff"

print("=" * 50)
print("PHISHGUARD AI - DATASET INSPECTION")
print("=" * 50)

if not DATA_FILE.exists():
    print("ERROR: Dataset file not found!")
    print("Expected location:", DATA_FILE)
    print("Please check the file name and folder.")
    raise SystemExit(1)

with open(DATA_FILE, "r", encoding="utf-8-sig") as file:
    dataset = arff.load(file)

df = pd.DataFrame(
    dataset["data"],
    columns=[item[0] for item in dataset["attributes"]]
)

print("\nDataset loaded successfully.")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print("-", column)

print("\nFirst five rows:")
print(df.head().to_string())

print("\nMissing values:", df.isna().sum().sum())

print("\nUnique values in each column:")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(10))

print("\nDataset inspection completed.")
