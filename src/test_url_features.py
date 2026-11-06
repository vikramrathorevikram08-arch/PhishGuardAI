
from url_feature_extractor import extract_url_features


test_urls = [
    "https://www.google.com",
    "http://192.168.1.10/login",
    "https://secure-account-verify.example.com/signin",
    "bit.ly/example"
]

for url in test_urls:
    print("\nURL:", url)

    try:
        features = extract_url_features(url)

        for name, value in features.items():
            print(f"{name}: {value}")

    except ValueError as error:
        print("Error:", error)

print("\nURL FEATURE EXTRACTION TEST COMPLETE")
