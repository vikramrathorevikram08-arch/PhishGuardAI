
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "phishing_clean.csv"
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("PHISHGUARD AI - ADVANCED EVALUATION")
print("=" * 55)

if not DATA_FILE.exists():
    raise FileNotFoundError("Prepared dataset is missing.")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["Result"])
y = df["Result"].astype(int)

# Use the same split settings as the training program.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model_files = sorted(MODEL_DIR.glob("*.joblib"))

if not model_files:
    raise FileNotFoundError(
        "No saved models found. Run src/train_models.py first."
    )

roc_results = []
importance_results = []

# -------------------------------------------------
# ROC CURVES
# -------------------------------------------------
plt.figure(figsize=(10, 7))

for model_file in model_files:
    artifact = joblib.load(model_file)

    if not isinstance(artifact, dict) or "model" not in artifact:
        print("Skipping unexpected model file:", model_file.name)
        continue

    model = artifact["model"]
    feature_columns = artifact.get("feature_columns")

    if feature_columns != list(X.columns):
        raise ValueError(
            f"Feature mismatch in {model_file.name}. "
            "Retrain the models on the current dataset."
        )

    print("\nEvaluating:", model_file.stem)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    auc = roc_auc_score(y_test, scores)
    fpr, tpr, _ = roc_curve(y_test, scores)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_file.stem} (AUC={auc:.3f})"
    )

    roc_results.append({
        "Model": model_file.stem,
        "ROC_AUC": auc
    })

    # Save one confusion-matrix image per model.
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Legitimate", "Phishing"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"Confusion Matrix: {model_file.stem}")
    fig.tight_layout()

    matrix_path = RESULT_DIR / f"confusion_{model_file.stem}.png"
    fig.savefig(matrix_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Permutation importance works with pipelines as well.
    # It measures the effect of shuffling each feature.
    if model_file.stem in [
        "random_forest",
        "decision_tree",
        "gradient_boosting"
    ]:
        importance = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=3,
            random_state=42,
            n_jobs=-1
        )

        for feature, value in zip(
            X.columns,
            importance.importances_mean
        ):
            importance_results.append({
                "Model": model_file.stem,
                "Feature": feature,
                "Importance": value
            })

plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - PhishGuard AI")
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

roc_path = RESULT_DIR / "roc_comparison.png"
plt.savefig(roc_path, dpi=200, bbox_inches="tight")
plt.close()

pd.DataFrame(roc_results).to_csv(
    RESULT_DIR / "roc_scores.csv",
    index=False
)

print("\nROC curve saved to:", roc_path)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
if importance_results:
    importance_df = pd.DataFrame(importance_results)

    importance_df.to_csv(
        RESULT_DIR / "feature_importance.csv",
        index=False
    )

    # Average permutation importance across selected tree models.
    average_importance = (
        importance_df.groupby("Feature")["Importance"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    average_importance.sort_values().plot(
        kind="barh",
        ax=ax
    )

    ax.set_title("Top 15 Features by Average Permutation Importance")
    ax.set_xlabel("Mean decrease in F1-score after shuffling")
    ax.set_ylabel("Website Feature")

    fig.tight_layout()

    importance_path = RESULT_DIR / "feature_importance.png"
    fig.savefig(importance_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Feature importance saved to:", importance_path)
else:
    print("No supported models available for feature importance.")

print("\nSaved evaluation files:")
for path in sorted(RESULT_DIR.glob("*.png")):
    print("-", path.name)

print("\nADVANCED EVALUATION COMPLETE")
