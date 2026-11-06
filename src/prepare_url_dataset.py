
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR / "data" / "raw"
    / "phishing+websites" / "url_data"
    / "phishing_url_dataset.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "url_dataset_clean.csv"


print("=" * 60)
print("PHISHGUARD AI - URL DATASET PREPARATION")
print("=" * 60)

# Check whether the source file exists.
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

# Read the CSV without assuming it has a header.
df = pd.read_csv(
    DATA_FILE,
    header=None,
    dtype=str,
    keep_default_na=False
)

print("\nOriginal dataset shape:", df.shape)

if df.shape[1] < 2:
    raise ValueError(
        "The CSV must contain a URL column and a label column."
    )

# If there are extra columns, combine all columns except
# the last one into the URL field.
if df.shape[1] > 2:
    url_values = df.iloc[:, :-1].astype(str).agg(",".join, axis=1)
    label_values = df.iloc[:, -1].astype(str)
    df = pd.DataFrame({
        "url": url_values,
        "label": label_values
    })
else:
    df.columns = ["url", "label"]

# Remove a header row if one exists.
first_url = str(df.iloc[0]["url"]).strip().lower()
first_label = str(df.iloc[0]["label"]).strip().lower()

url_headers = {"url", "urls", "website", "link", "domain"}
label_headers = {"label", "class", "result", "type", "status"}

if first_url in url_headers and first_label in label_headers:
    df = df.iloc[1:].copy()

# Clean whitespace and standardize label text.
df["url"] = df["url"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Remove empty records.
df = df[
    (df["url"] != "") &
    (df["label"] != "")
].copy()

# Keep only the expected binary labels.
df["label"] = df["label"].str.lower()

valid_labels = {"0", "1"}
invalid_rows = ~df["label"].isin(valid_labels)

if invalid_rows.any():
    print("\nUnexpected labels found:")
    print(df.loc[invalid_rows, "label"].value_counts().to_string())
    raise ValueError(
        "Labels must be 0 or 1. Check the dataset before continuing."
    )

df["label"] = df["label"].astype(int)

print("\nLabel counts in the original dataset:")
print(df["label"].value_counts().sort_index().to_string())

# This dataset's expected convention:
# 0 = phishing
# 1 = legitimate
df["is_phishing"] = (df["label"] == 0).astype(int)

# Remove duplicate URLs, ignoring capitalization.
df["url_key"] = df["url"].str.lower()

before_duplicates = len(df)
df = df.drop_duplicates(subset="url_key", keep="first").copy()
duplicates_removed = before_duplicates - len(df)

df = df[["url", "is_phishing"]].reset_index(drop=True)

# Confirm both classes are present.
if df["is_phishing"].nunique() != 2:
    raise ValueError(
        "The cleaned dataset must contain both phishing and "
        "legitimate examples."
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nPreparation summary")
print("-" * 40)
print("Rows after cleaning:", len(df))
print("Duplicate URLs removed:", duplicates_removed)
print("Phishing URLs:", int((df["is_phishing"] == 1).sum()))
print("Legitimate URLs:", int((df["is_phishing"] == 0).sum()))
print("Output file:", OUTPUT_FILE)

print("\nFirst five cleaned records:")
print(df.head().to_string(index=False))

print("\nURL DATASET PREPARATION COMPLETE")
