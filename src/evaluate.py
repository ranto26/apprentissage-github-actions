import json
import os
import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import load_data


def evaluate_model(model_path=Path("models/iris_model.joblib")):

    # 1. Charger le modele sauvegarde
    model = joblib.load(model_path)

    # 2. Recuperer uniquement les donnees d'evaluation
    _, X_test, _, y_test = load_data()

    # 3. Generer les predictions
    predictions = model.predict(X_test)

    # 4. Calculer les metriques
    metrics = {
        "accuracy": accuracy_score(y_true=y_test, y_pred=predictions),
        "precision": precision_score(
            y_true=y_test, y_pred=predictions, average="macro", zero_division=0
        ),
        "recall": recall_score(
            y_true=y_test, y_pred=predictions, average="macro", zero_division=0
        ),
        "f1_score": f1_score(
            y_true=y_test, y_pred=predictions, average="macro", zero_division=0
        ),
    }

    # 5. Afficher les metriques
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    return metrics


def save_metrics(
    metrics,
    report_path=Path("reports/metrics.json"),
):
    """Enregistre les metriques dans un fichier JSON."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Metriques enregistrees : {report_path}")


def validate_threshold(min_accuracy):
    """Verifie la validite du seuil."""

    if not 0 <= min_accuracy <= 1:
        raise ValueError("Le seuil doit être compris entre 0 et 1.")


def validate_model(metrics, min_accuracy=0.90):
    """Verifie que le modele respecte le seuil minimal."""

    validate_threshold(min_accuracy)

    accuracy = metrics["accuracy"]

    print(f"Seuil minimal : {min_accuracy:.2%}")

    if accuracy >= min_accuracy:
        print(f"Modele valide : {accuracy:.2%}")
        return True

    print(f"Modele rejete : {accuracy:.2%}")
    return False


def main():
    min_accuracy = float(os.getenv("MIN_ACCURACY", "0.90"))

    validate_threshold(min_accuracy)

    metrics = evaluate_model()

    save_metrics(metrics)

    is_valid = validate_model(metrics, min_accuracy)

    return 0 if is_valid else 1


if __name__ == "__main__":
    sys.exit(main())
