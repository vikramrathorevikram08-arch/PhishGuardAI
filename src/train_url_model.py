
from pathlib import Path
import json
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)

from url_feature_extractor import extract_url_features


PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "url_dataset_clean.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 65)
print("PHISHGUARD AI - URL MODEL TRAINING")
print("=" * 65)


# Step 1: Load the cleaned dataset.
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_FILE}"
    )

df = pd.read_csv(DATA_FILE)

required_columns = {"url", "is_phishing"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "Dataset must contain 'url' and 'is_phishing' columns."
    )

df = df.dropna(subset=["url", "is_phishing"]).copy()

df["url"] = df["url"].astype(str).str.strip()

df["is_phishing"] = pd.to_numeric(
    df["is_phishing"],
    errors="coerce"
)

df = df.dropna(subset=["is_phishing"]).copy()

if not df["is_phishing"].isin([0, 1]).all():
    raise ValueError("Labels must be 0 or 1.")

df["is_phishing"] = df["is_phishing"].astype(int)

df = df[df["url"] != ""].copy()

df["url_key"] = df["url"].str.lower()
df = df.drop_duplicates(subset="url_key").reset_index(drop=True)
df = df.drop(columns=["url_key"])

print("\nDataset rows:", len(df))
print(
    "Phishing (1):",
    int((df["is_phishing"] == 1).sum())
)
print(
    "Legitimate (0):",
    int((df["is_phishing"] == 0).sum())
)

if df["is_phishing"].nunique() != 2:
    raise ValueError("Both classes must be present.")


# Step 2: Extract features from URLs.
print("\nExtracting URL features. Please wait...")

feature_rows = []
valid_labels = []
invalid_count = 0

for _, row in df.iterrows():
    try:
        features = extract_url_features(row["url"])

        feature_rows.append(features)
        valid_labels.append(int(row["is_phishing"]))

    except (ValueError, TypeError) as error:
        invalid_count += 1

X = pd.DataFrame(feature_rows).replace(
    [np.inf, -np.inf],
    np.nan
)

y = pd.Series(valid_labels, name="is_phishing")

# Remove rows containing unusable feature values.
valid_rows = X.notna().all(axis=1)

X = X.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

print("Valid URLs used:", len(X))
print("Invalid URLs skipped:", invalid_count)
print("Feature count:", X.shape[1])

if len(X) < 20:
    raise ValueError(
        "Too few valid URLs. Check the dataset and feature extractor."
    )

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "Not enough examples of both classes to train and test."
    )


# Step 3: Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# Step 4: Define five models.
# SVM is temporarily excluded because it was too slow.
models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "decision_tree": DecisionTreeClassifier(
        max_depth=20,
        class_weight="balanced",
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),

    "k_nearest_neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ))
    ])
}


# Step 5: Train and evaluate each model.
comparison = []
fitted_models = {}

plt.figure(figsize=(9, 7))

for name, model in models.items():

    print(f"\nTraining: {name}", flush=True)

    start_time = time.time()

    model.fit(X_train, y_train)

    training_seconds = time.time() - start_time

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    comparison.append({
        "model": name,
        "accuracy": accuracy,
        "precision_phishing": precision,
        "recall_phishing": recall,
        "f1_phishing": f1,
        "roc_auc": auc,
        "training_seconds": training_seconds,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    })

    model_path = MODEL_DIR / f"url_{name}.joblib"

    joblib.dump(model, model_path)

    fitted_models[name] = model

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Saved: {model_path}")

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=name
    )


# Step 6: Save ROC comparison graph.
plt.title("PhishGuard AI - URL Model ROC Comparison")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "url_roc_comparison.png",
    dpi=150
)

plt.close()


# Step 7: Save model comparison results.
results_df = pd.DataFrame(comparison)

results_df = results_df.sort_values(
    by=["f1_phishing", "recall_phishing"],
    ascending=False
).reset_index(drop=True)

results_df.to_csv(
    RESULTS_DIR / "url_model_comparison.csv",
    index=False
)


# Step 8: Select and save the best model.
best_name = results_df.iloc[0]["model"]
best_model = fitted_models[best_name]

joblib.dump(
    best_model,
    MODEL_DIR / "url_best_model.joblib"
)


# Step 9: Save model metadata.
metadata = {
    "best_model": best_name,
    "target": "is_phishing",
    "label_1": "phishing",
    "label_0": "legitimate",
    "features": list(X.columns),
    "test_size": 0.20,
    "random_state": 42,
    "training_rows": len(X_train),
    "testing_rows": len(X_test),
    "invalid_urls_skipped": invalid_count
}

with open(
    MODEL_DIR / "url_model_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)


# Step 10: Display final results.
print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

print("\nBest model by phishing F1-score:", best_name)

print("\nClassification report for the best model:")

print(
    classification_report(
        y_test,
        best_model.predict(X_test),
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    )
)

print("\nSaved files:")
print("models/url_best_model.joblib")
print("models/url_model_metadata.json")
print("results/url_model_comparison.csv")
print("results/url_roc_comparison.png")

print("\nURL MODEL TRAINING COMPLETE")
