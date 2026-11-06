
from pathlib import Path
import json
import re
import ipaddress
from urllib.parse import urlsplit, parse_qs

import joblib
import pandas as pd

try:
    from .url_feature_extractor import extract_url_features
except ImportError:
    from url_feature_extractor import extract_url_features


PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_DIR / "models" / "url_best_model.joblib"
METADATA_FILE = PROJECT_DIR / "models" / "url_model_metadata.json"

SUSPICIOUS_WORDS = [
    "login", "verify", "verification", "account", "secure",
    "update", "password", "signin", "confirm", "banking",
    "wallet", "suspended", "recover", "payment", "credential"
]

SHORTENING_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd",
    "cutt.ly", "ow.ly", "buff.ly", "shorturl.at"
}


def get_url_indicators(url):
    """Calculate simple, explainable indicators from the submitted URL."""

    candidate = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", candidate):
        candidate = "https://" + candidate

    parsed = urlsplit(candidate)
    hostname = (parsed.hostname or "").lower()
    full_url = url.strip().lower()
    path_and_query = (parsed.path + "?" + parsed.query).lower()

    try:
        ipaddress.ip_address(hostname)
        is_ip = bool(hostname)
    except ValueError:
        is_ip = False

    labels = [part for part in hostname.split(".") if part]
    subdomain_count = max(0, len(labels) - 2)

    try:
        port = parsed.port
    except ValueError:
        port = None

    query_params = parse_qs(parsed.query)
    redirect_parameter = any(
        key.lower() in {
            "redirect", "redirect_url", "redirect_uri",
            "return", "returnurl", "url", "next", "continue",
            "target", "dest", "destination"
        }
        for key in query_params
    )

    suspicious_word = any(
        word in full_url for word in SUSPICIOUS_WORDS
    )

    shortened = (
        hostname in SHORTENING_DOMAINS
        or any(hostname.endswith("." + domain)
               for domain in SHORTENING_DOMAINS)
    )

    indicators = [
        ("Uses HTTPS", parsed.scheme.lower() == "https",
         "HTTPS is enabled", "Does not use HTTPS"),
        ("Hostname is an IP address", is_ip,
         "Hostname is an IP address", "Hostname is not an IP address"),
        ("URL length is over 75 characters", len(url) > 75,
         f"URL length is {len(url)} characters",
         f"URL length is {len(url)} characters"),
        ("URL contains an @ symbol", "@" in url,
         "URL contains an @ symbol", "Does not contain an @ symbol"),
        ("Hostname contains multiple hyphens",
         hostname.count("-") >= 2,
         "Hostname contains multiple hyphens",
         "Hostname does not contain multiple hyphens"),
        ("Hostname contains many subdomain levels",
         subdomain_count >= 3,
         f"Hostname has {subdomain_count} subdomain level(s)",
         f"Hostname has {subdomain_count} subdomain level(s)"),
        ("Uses a known URL shortening domain", shortened,
         "Uses a known URL shortening domain",
         "Does not use a known URL shortening domain"),
        ("Contains a login/account-related keyword",
         suspicious_word,
         "Contains a login/account-related keyword",
         "No checked login/account-related keyword detected"),
        ("Contains a non-standard port",
         port is not None and port not in (80, 443),
         f"Uses port {port}",
         "No non-standard port detected"),
        ("Contains a redirect-like parameter", redirect_parameter,
         "Contains a redirect-like parameter",
         "No checked redirect-like parameter detected"),
    ]

    return [
        {
            "label": label,
            "detected": bool(detected),
            "message": positive if detected else negative,
        }
        for label, detected, positive, negative in indicators
    ]


class URLPredictor:

    def __init__(self):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_FILE}"
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Model metadata not found: {METADATA_FILE}"
            )

        self.model = joblib.load(MODEL_FILE)

        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.feature_names = self.metadata["features"]
        self.model_name = self.metadata["best_model"]

        if not self.feature_names:
            raise ValueError("Model feature list is empty.")

    def predict_url(self, url):

        if not isinstance(url, str) or not url.strip():
            raise ValueError("Please enter a URL.")

        url = url.strip()

        if re.match(
            r"^(javascript|data|file|ftp):",
            url,
            flags=re.IGNORECASE
        ):
            raise ValueError(
                "Enter a website URL using HTTP or HTTPS."
            )

        # Add a scheme for parsing if the user entered only a domain.
        parse_candidate = url
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
            parse_candidate = "https://" + url

        parsed = urlsplit(parse_candidate)

        if not parsed.hostname or "." not in parsed.hostname:
            raise ValueError(
                "Enter a valid website URL, such as https://example.com."
            )

        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("Only HTTP and HTTPS URLs are supported.")

        # Use the same feature extraction used during model training.
        features = extract_url_features(url)
        input_data = pd.DataFrame([features])
        input_data = input_data.reindex(columns=self.feature_names)

        if input_data.isnull().any().any():
            raise ValueError(
                "Could not calculate all required URL features."
            )

        prediction = int(self.model.predict(input_data)[0])

        result = {
            "url": url,
            "prediction": prediction,
            "classification": (
                "Likely Phishing"
                if prediction == 1
                else "Likely Legitimate"
            ),
            "model": self.model_name,
            "feature_count": len(self.feature_names),
            "phishing_probability": None,
            "legitimate_probability": None,
            "model_score_percent": None,
            "indicators": get_url_indicators(url),
            "warning": (
                "This is an ML estimate, not proof that a website "
                "is safe or malicious."
            ),
        }

        # Labels in this project: 1 = phishing, 0 = legitimate.
        # Only show both probabilities if the model provides both classes.
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(input_data)[0]
            classes = list(self.model.classes_)

            if 1 in classes:
                phishing_probability = float(
                    probabilities[classes.index(1)]
                )
                result["phishing_probability"] = phishing_probability
                result["model_score_percent"] = round(
                    phishing_probability * 100, 2
                )

            if 0 in classes:
                result["legitimate_probability"] = float(
                    probabilities[classes.index(0)]
                )

        return result


if __name__ == "__main__":

    print("=" * 55)
    print("PHISHGUARD AI - URL PREDICTION")
    print("=" * 55)

    try:
        predictor = URLPredictor()
        print("Loaded model:", predictor.model_name)

        while True:
            url = input(
                "\nEnter a URL (or type exit to stop): "
            ).strip()

            if url.lower() == "exit":
                break

            try:
                result = predictor.predict_url(url)

                print("\nPrediction result")
                print("-" * 40)
                print("URL:", result["url"])
                print("Classification:", result["classification"])
                print("Model:", result["model"])
                print("Features used:", result["feature_count"])

                if result["phishing_probability"] is not None:
                    print(
                        "Estimated phishing probability:",
                        f'{result["phishing_probability"] * 100:.2f}%'
                    )

                if result["legitimate_probability"] is not None:
                    print(
                        "Estimated legitimate probability:",
                        f'{result["legitimate_probability"] * 100:.2f}%'
                    )

                print("Warning:", result["warning"])

            except ValueError as error:
                print("Input error:", error)

    except (FileNotFoundError, KeyError, OSError) as error:
        print("Could not load the prediction model:", error)

    print("\nURL PREDICTION PROGRAM CLOSED")
