"""
Basic smoke tests for the API. Requires the model files to be
resolvable (either on Kaggle or in ./models locally) since
SkinLesionPredictor loads the real model on startup.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "device" in response.json()


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "val_accuracy" in response.json()


def test_predict_missing_fields_returns_422():
    # No image/age/sex/localization supplied — FastAPI should reject
    # this as a validation error before it ever reaches the model.
    response = client.post("/predict")
    assert response.status_code == 422
