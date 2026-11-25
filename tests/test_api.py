from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_setosa():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["species"] == "setosa"


def test_invalid_input():
    payload = {
        "sepal_length": -5,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
