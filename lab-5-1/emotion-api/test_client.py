import json
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))


def test_single_prediction():
    data = {
        "text": "I am feeling absolutely wonderful today!",
        "model_version": "default",
    }
    response = requests.post(f"{BASE_URL}/predict/emotion", json=data)
    print("\nSingle Prediction:")
    print(json.dumps(response.json(), indent=2))


def test_batch_prediction():
    texts = [
        "This is amazing news!",
        "I am very angry about this situation",
        "I feel scared and anxious",
        "This makes me so happy",
    ]
    response = requests.post(f"{BASE_URL}/predict/emotion/batch", json=texts)
    print("\nBatch Prediction:")
    print(json.dumps(response.json(), indent=2))


def test_invalid_request():
    data = {
        "text": ""  # Пустой текст
    }
    # Invalid single prediction request
    response = requests.post(f"{BASE_URL}/predict/emotion", json=data)
    print("\nInvalid Request (single):")
    print(f"Status: {response.status_code}")
    try:
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    print("Testing Emotion Classification API")
    print("=" * 50)
    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_invalid_request()
