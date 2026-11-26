from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data import load_data


def train_model(model_path=Path("models/iris_model.joblib")):

    X_train, X_test, y_train, y_test = load_data()

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Modele enregistre : {model_path}")


if __name__ == "__main__":
    train_model()
