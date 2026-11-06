
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

from src.url_predictor import URLPredictor


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    PROJECT_DIR / "data" / "processed"
    / "phishing_clean.csv"
)

RESULT_FILE = (
    PROJECT_DIR / "results"
    / "model_comparison.csv"
)

MODEL_DIR = PROJECT_DIR / "models"


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ PhishGuard AI")

st.subheader(
    "Phishing Website Detection Using Machine Learning"
)

st.write(
    "Compare machine-learning models, test website features, "
    "and analyze URLs using a separate URL-based model."
)


# --------------------------------------------------
# LOAD ORIGINAL DATASET AND RESULTS
# --------------------------------------------------

if not DATA_FILE.exists() or not RESULT_FILE.exists():
    st.error("Required original dataset or results file is missing.")
    st.info(
        "Run these commands from the project directory:\n\n"
        "1. `python src/prepare_data.py`\n"
        "2. `python src/train_models.py`"
    )
    st.stop()

df = pd.read_csv(DATA_FILE)
results = pd.read_csv(RESULT_FILE)

if "Result" not in df.columns:
    st.error("The original dataset does not contain the Result target column.")
    st.stop()

feature_columns = [
    column for column in df.columns
    if column != "Result"
]


# --------------------------------------------------
# MODEL LOADER
# --------------------------------------------------

@st.cache_resource
def load_url_predictor():
    return URLPredictor()


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Overview",
        "Model Comparison",
        "Test a Prediction",
        "URL Phishing Detector"
    ]
)


# ==================================================
# PAGE 1: OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Project Overview")

    total_rows = len(df)
    total_features = len(feature_columns)

    phishing_count = int((df["Result"] == 1).sum())
    legitimate_count = int((df["Result"] == 0).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Records", f"{total_rows:,}")
    col2.metric("Website Features", total_features)
    col3.metric("Phishing Records", f"{phishing_count:,}")
    col4.metric("Legitimate Records", f"{legitimate_count:,}")

    st.markdown("### Dataset class distribution")

    class_data = pd.DataFrame({
        "Website Type": ["Phishing", "Legitimate"],
        "Records": [phishing_count, legitimate_count]
    }).set_index("Website Type")

    st.bar_chart(class_data)

    st.markdown("### Project workflow")

    st.write(
        "Dataset preparation → Model training → Evaluation → "
        "Model comparison → URL feature extraction → URL prediction"
    )

    st.markdown("### Original model input features")
    st.write(", ".join(feature_columns))

    st.info(
        "The original model uses pre-extracted website features. "
        "The URL Phishing Detector uses a separate model trained "
        "on URL-derived features."
    )


# ==================================================
# PAGE 2: MODEL COMPARISON
# ==================================================

elif page == "Model Comparison":

    st.header("Model Performance Comparison")

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]

    missing_metrics = [
        column for column in metric_columns
        if column not in results.columns
    ]

    if missing_metrics:
        st.error("Missing metric columns: " + ", ".join(missing_metrics))
        st.stop()

    display_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "Training_Time_Seconds"
    ]

    display_columns = [
        column for column in display_columns
        if column in results.columns
    ]

    st.dataframe(
        results[display_columns].style.format({
            column: "{:.4f}"
            for column in metric_columns
            if column in display_columns
        }),
        use_container_width=True,
        hide_index=True
    )

    selected_metric = st.selectbox(
        "Choose a metric to compare",
        metric_columns,
        index=3
    )

    chart_data = results[
        ["Model", selected_metric]
    ].set_index("Model")

    st.bar_chart(chart_data)

    best_row = results.loc[results[selected_metric].idxmax()]

    st.success(
        f"Best model by {selected_metric}: "
        f"{best_row['Model']} "
        f"({best_row[selected_metric]:.4f})"
    )

    st.caption(
        "These are saved evaluation results. They do not guarantee "
        "performance on previously unseen websites."
    )

    st.markdown("### Confusion matrix counts")

    confusion_columns = [
        "True_Negatives",
        "False_Positives",
        "False_Negatives",
        "True_Positives"
    ]

    if all(column in results.columns for column in confusion_columns):

        selected_model = st.selectbox(
            "Select model",
            results["Model"].tolist(),
            key="confusion_model"
        )

        row = results[
            results["Model"] == selected_model
        ].iloc[0]

        matrix = pd.DataFrame(
            [
                [
                    int(row["True_Negatives"]),
                    int(row["False_Positives"])
                ],
                [
                    int(row["False_Negatives"]),
                    int(row["True_Positives"])
                ]
            ],
            index=["Actual legitimate", "Actual phishing"],
            columns=["Predicted legitimate", "Predicted phishing"]
        )

        st.dataframe(matrix, use_container_width=True)


# ==================================================
# PAGE 3: ORIGINAL WEBSITE-FEATURE PREDICTION
# ==================================================

elif page == "Test a Prediction":

    st.header("Test a Website-Feature Record")

    if "Model" not in results.columns:
        st.error("The original model comparison file has no Model column.")
        st.stop()

    model_names = results["Model"].dropna().tolist()

    selected_model = st.selectbox(
        "Choose a trained model",
        model_names
    )

    model_file = (
        selected_model.lower().replace(" ", "_") + ".joblib"
    )

    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        st.error(f"Model file not found: {model_path.name}")
        st.stop()

    artifact = joblib.load(model_path)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        saved_features = artifact.get("feature_columns")

        if model is None or saved_features is None:
            st.error(
                "The saved model artifact does not contain "
                "'model' and 'feature_columns'."
            )
            st.stop()
    else:
        st.error(
            "Unexpected model file format. Expected a dictionary "
            "containing model and feature_columns."
        )
        st.stop()

    if list(saved_features) != feature_columns:
        st.error(
            "Feature columns do not match the saved model. "
            "Retrain the original models using the current dataset."
        )
        st.stop()

    st.write(
        "Choose the feature values below. These values represent "
        "website characteristics, not URL text."
    )

    st.info(
        "For a meaningful test, use feature values from the prepared "
        "dataset or values whose meanings you understand."
    )

    with st.form("prediction_form"):

        st.markdown("### Website feature values")

        input_values = {}
        layout_columns = st.columns(3)

        for index, feature in enumerate(feature_columns):

            numeric_values = pd.to_numeric(
                df[feature], errors="coerce"
            ).dropna()

            options = sorted(numeric_values.unique().tolist())

            if not options:
                st.error(f"No numeric values available for {feature}.")
                st.stop()

            with layout_columns[index % 3]:
                input_values[feature] = st.selectbox(
                    feature,
                    options,
                    key=f"feature_{feature}"
                )

        submitted = st.form_submit_button("Analyze Feature Record")

    if submitted:

        input_df = pd.DataFrame(
            [[input_values[column] for column in saved_features]],
            columns=saved_features
        )

        prediction = int(model.predict(input_df)[0])

        st.markdown("### Prediction result")

        if prediction == 1:
            st.error("Predicted class: PHISHING")
        else:
            st.success("Predicted class: LEGITIMATE")

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                phishing_probability = float(
                    probabilities[classes.index(1)]
                )

                st.metric(
                    "Estimated phishing probability",
                    f"{phishing_probability:.2%}"
                )

        st.caption(
            "This prediction is based on manually selected dataset "
            "features, not a direct scan of a live website."
        )


# ==================================================
# PAGE 4: URL PHISHING DETECTOR
# ==================================================

elif page == "URL Phishing Detector":

    st.header("URL Phishing Detector")

    st.write(
        "Enter a website URL to analyze its URL-derived features "
        "using your separately trained URL machine-learning model."
    )

    st.info(
        "This page performs offline URL analysis. It does not "
        "automatically visit the website or check live reputation."
    )

    with st.form("url_prediction_form"):

        entered_url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
        )

        analyze_url = st.form_submit_button(
            "Analyze URL",
            type="primary"
        )

    if analyze_url:

        if not entered_url.strip():
            st.warning("Please enter a URL.")

        else:

            try:
                with st.spinner("Analyzing URL characteristics..."):
                    predictor = load_url_predictor()
                    result = predictor.predict_url(entered_url.strip())

                st.markdown("---")
                st.markdown("## Prediction Result")

                st.code(result["url"], language=None)

                if result["prediction"] == 1:
                    st.error(
                        "🚨 Model prediction: Likely phishing URL"
                    )
                else:
                    st.success(
                        "✅ Model prediction: Likely legitimate URL"
                    )

                phishing_probability = result.get(
                    "phishing_probability"
                )
                legitimate_probability = result.get(
                    "legitimate_probability"
                )

                # Display probability bars only when the model
                # supplies the corresponding class probabilities.
                if phishing_probability is not None:

                    st.markdown("### Estimated phishing probability")

                    st.progress(
                        max(0.0, min(1.0, phishing_probability))
                    )

                    st.write(f"**{phishing_probability * 100:.2f}%**")

                if legitimate_probability is not None:

                    st.markdown("### Estimated legitimate probability")

                    st.progress(
                        max(0.0, min(1.0, legitimate_probability))
                    )

                    st.write(f"**{legitimate_probability * 100:.2f}%**")

                st.caption(
                    "These are model-estimated probabilities, not "
                    "verified risk scores. They do not guarantee "
                    "the website's safety."
                )

                # Expandable URL indicators
                with st.expander("View basic URL indicators"):

                    for item in result["indicators"]:

                        if item["detected"]:
                            st.markdown(
                                f"⚠️ **Detected:** {item['message']}"
                            )
                        else:
                            st.markdown(
                                f"✓ **Not detected:** {item['message']}"
                            )

                # Model details
                st.markdown("### Model details")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Model used",
                    result["model"]
                )

                col2.metric(
                    "Features analyzed",
                    result["feature_count"]
                )

                st.markdown("### Important safety note")

                st.warning(
                    "This is an ML estimate, not proof that a website "
                    "is safe or malicious."
                )

                st.write(
                    "A URL can look normal and still lead to a phishing "
                    "page. Conversely, a suspicious-looking URL may be "
                    "legitimate. Do not enter passwords or payment details "
                    "based only on this prediction."
                )

                st.info(
                    "The prediction uses URL text only. It does not "
                    "inspect webpage content, follow redirects, or verify "
                    "the website's current reputation."
                )

            except (
                ValueError,
                FileNotFoundError,
                OSError,
                KeyError
            ) as error:

                st.error(f"Could not analyze the URL: {error}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "PhishGuard AI | Machine Learning Project | "
    "Predictions are estimates, not security guarantees."
)
