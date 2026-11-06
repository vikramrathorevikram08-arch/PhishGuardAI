
from urllib.parse import urlparse
from collections import Counter
import ipaddress
import math
import re


SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "account",
    "update",
    "secure",
    "signin",
    "password",
    "confirm",
    "banking",
    "wallet",
    "payment",
    "suspend",
    "recover",
    "authenticate",
    "webscr",
    "credential",
    "unlock"
]

SHORTENING_SERVICES = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "buff.ly",
    "ow.ly",
    "cutt.ly",
    "rebrand.ly",
    "shorturl.at"
]


def extract_url_features(url):
    """
    Extract numerical and binary features from a URL.
    These features are used by the URL-based ML model.
    """

    url = str(url).strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    # Add a scheme only when one is missing.
    normalized_url = url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        normalized_url = "http://" + url

    parsed = urlparse(normalized_url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    if not hostname:
        raise ValueError("Please enter a valid URL with a hostname.")

    lower_url = normalized_url.lower()
    lower_host = hostname.lower()

    try:
        ipaddress.ip_address(lower_host)
        has_ip_address = 1
    except ValueError:
        has_ip_address = 0

    try:
        port = parsed.port
        has_nonstandard_port = int(port is not None and port not in (80, 443))
    except ValueError:
        has_nonstandard_port = 1

    hostname_parts = lower_host.split(".")
    suspicious_word_count = sum(
        lower_url.count(word) for word in SUSPICIOUS_WORDS
    )

    digit_count = sum(character.isdigit() for character in normalized_url)
    letter_count = sum(character.isalpha() for character in normalized_url)

    special_characters = sum(
        not character.isalnum() for character in normalized_url
    )

    entropy_counts = Counter(normalized_url)
    url_length = len(normalized_url)
    entropy = 0.0

    if url_length:
        entropy = -sum(
            (count / url_length) * math.log2(count / url_length)
            for count in entropy_counts.values()
        )

    features = {
        "url_length": url_length,
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "count_dots": normalized_url.count("."),
        "count_hyphens": normalized_url.count("-"),
        "count_at": normalized_url.count("@"),
        "count_question_marks": normalized_url.count("?"),
        "count_equal": normalized_url.count("="),
        "count_ampersand": normalized_url.count("&"),
        "count_percent": normalized_url.count("%"),
        "count_slashes": normalized_url.count("/"),
        "count_digits": digit_count,
        "count_letters": letter_count,
        "count_special_characters": special_characters,
        "has_ip_address": has_ip_address,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_nonstandard_port": has_nonstandard_port,
        "subdomain_count": max(0, len(hostname_parts) - 2),
        "has_suspicious_word": int(suspicious_word_count > 0),
        "suspicious_word_count": suspicious_word_count,
        "uses_shortener": int(
            any(
                lower_host == service
                or lower_host.endswith("." + service)
                for service in SHORTENING_SERVICES
            )
        ),
        "has_double_slash_in_path": int("//" in path),
        "digit_ratio": digit_count / max(1, url_length),
        "special_character_ratio": special_characters / max(1, url_length),
        "url_entropy": entropy,
    }

    return features
