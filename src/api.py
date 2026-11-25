import os
from functools import lru_cache
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Prediction API")

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@lru_cache
def load_model():
    model_path = Path(os.getenv("MODEL_PATH", "models/iris_model.joblib"))
    return joblib.load(model_path)


@app.get("/health")
def health():
    load_model()
    return {"status": "ok"}


@app.post("/predict")
def predict(features: IrisFeatures):
    values = [
        [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]
    ]

    prediction = int(load_model().predict(values)[0])

    return {
        "class_id": prediction,
        "species": CLASS_NAMES[prediction],
    }
