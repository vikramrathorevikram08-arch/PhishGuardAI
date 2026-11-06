
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "phishing_clean.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("PHISHGUARD AI - MODEL TRAINING AND COMPARISON")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Prepared dataset not found: {DATA_FILE}\n"
        "Run python src/prepare_data.py first."
    )

df = pd.read_csv(DATA_FILE)

if "Result" not in df.columns:
    raise ValueError("Target column 'Result' is missing.")

if df.isna().any().any():
    raise ValueError("Dataset contains missing values.")

X = df.drop(columns=["Result"])
y = df["Result"].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Expected labels: 1=phishing, 0=legitimate.")

# Keep the same split for all models to make comparisons fair.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset rows:", len(df))
print("Input features:", X.shape[1])
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Label convention: 1=phishing, 0=legitimate")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
            class_weight="balanced"
        ))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ])
}

results = []
best_model_name = None
best_model = None
best_f1 = -1.0

for name, model in models.items():
    print("\n" + "-" * 50)
    print("Training:", name)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start_time

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probabilities = model.decision_function(X_test)
    else:
        probabilities = None

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    auc = (
        roc_auc_score(y_test, probabilities)
        if probabilities is not None else float("nan")
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Training time: {training_seconds:.2f} seconds")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Phishing"],
        zero_division=0
    ))

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Training_Time_Seconds": training_seconds,
        "True_Negatives": int(tn),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp)
    })

    # Select the model by phishing-class F1 score.
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

    # Save each fitted model and its expected input columns.
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "label_mapping": {
                "1": "Phishing",
                "0": "Legitimate"
            }
        },
        MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib")
    )

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1_Score", ascending=False
)

csv_file = RESULT_DIR / "model_comparison.csv"
json_file = RESULT_DIR / "model_comparison.json"

results_df.to_csv(csv_file, index=False)
results_df.to_json(json_file, orient="records", indent=2)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results_df.round(4).to_string(index=False))

print("\nBest model by phishing F1-score:", best_model_name)
print(f"Best phishing F1-score: {best_f1:.4f}")

print("\nSaved comparison CSV:", csv_file)
print("Saved comparison JSON:", json_file)
print("Saved model files in:", MODEL_DIR)

print("\nMODEL TRAINING COMPLETE")
