
from pathlib import Path
import arff
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR / "data" / "raw"
    / "phishing+websites" / "Training Dataset.arff"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "phishing_clean.csv"

print("=" * 55)
print("PHISHGUARD AI - DATA PREPARATION")
print("=" * 55)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

with open(DATA_FILE, "r", encoding="utf-8-sig") as file:
    dataset = arff.load(file)

columns = [item[0] for item in dataset["attributes"]]
df = pd.DataFrame(dataset["data"], columns=columns)

print("\nOriginal rows:", len(df))
print("Original columns:", len(df.columns))

# Convert feature values and labels to numeric values.
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="raise")

# Confirm the expected target column and label values.
if "Result" not in df.columns:
    raise ValueError("Target column 'Result' was not found.")

labels = set(df["Result"].unique())

if labels != {-1, 1}:
    raise ValueError(
        f"Unexpected target labels: {labels}. "
        "Check the dataset documentation before proceeding."
    )

# Standard dataset convention:
# Result -1 = phishing; Result 1 = legitimate.
df["Result"] = df["Result"].map({
    -1: 1,
     1: 0
})

feature_columns = [
    column for column in df.columns if column != "Result"
]

print("\nOriginal class counts:")
print("Phishing (1):", int((df["Result"] == 1).sum()))
print("Legitimate (0):", int((df["Result"] == 0).sum()))

# Remove exact duplicate records.
exact_duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

print("\nExact duplicate rows removed:", exact_duplicates)

# Find identical feature patterns with conflicting labels.
label_counts = df.groupby(feature_columns)["Result"].nunique()
conflicting_patterns = label_counts[label_counts > 1].index

if len(conflicting_patterns) > 0:
    print(
        "Conflicting feature patterns found:",
        len(conflicting_patterns)
    )

    feature_index = pd.MultiIndex.from_frame(df[feature_columns])
    conflict_index = pd.MultiIndex.from_tuples(
        list(conflicting_patterns),
        names=feature_columns
    )

    ambiguous_mask = feature_index.isin(conflict_index)
    ambiguous_rows = int(ambiguous_mask.sum())

    print("Ambiguous rows excluded:", ambiguous_rows)
    df = df.loc[~ambiguous_mask].copy()
else:
    print("Conflicting feature patterns found: 0")

# Remove repeated feature vectors with the same label.
repeated_features = int(df.duplicated(subset=feature_columns).sum())
df = df.drop_duplicates(subset=feature_columns).copy()

print("Additional repeated feature rows removed:", repeated_features)

# Validate the final dataset.
if df.isna().any().any():
    raise ValueError("Missing values remain in the prepared dataset.")

if df[feature_columns].nunique().eq(1).any():
    constant_columns = df[feature_columns].columns[
        df[feature_columns].nunique().eq(1)
    ].tolist()
    print("Constant feature columns:", constant_columns)

if df["Result"].nunique() != 2:
    raise ValueError("Both target classes must remain in the dataset.")

df.to_csv(OUTPUT_FILE, index=False)

print("\nFINAL DATASET")
print("Rows:", len(df))
print("Input features:", len(feature_columns))
print("Columns including target:", len(df.columns))

print("\nFinal class distribution:")
print(df["Result"].value_counts().sort_index().to_string())

print("\nSaved clean dataset to:")
print(OUTPUT_FILE)

print("\nDATA PREPARATION COMPLETE")
