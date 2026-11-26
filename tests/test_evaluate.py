import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier

import src.evaluate as evaluation
from src.data import load_data
from src.evaluate import (
    evaluate_model,
    save_metrics,
    validate_model,
    validate_threshold,
)
from src.train import train_model


@pytest.mark.parametrize(
    "accuracy, min_accuracy, expected",
    [
        (0.95, 0.90, True),
        (0.85, 0.90, False),
        (0.9, 0.9, True),
        (1, 1, True),
        (0, 0, True),
        (0, 0.90, False),
        (0.99, 1, False),
        # Ajouter les autres scenarios
    ],
)
def test_validate_model(accuracy, min_accuracy, expected):
    # 1. Construire le dictionnaire de metriques
    metrics = {"accuracy": accuracy}
    # 2. Appeler la fonction a tester
    result = validate_model(metrics, min_accuracy)

    # 3. Verifier le resultat
    assert result is expected


@pytest.mark.parametrize(
    "min_accuracy",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_threshold_invalid(min_accuracy):
    with pytest.raises(ValueError, match=r"Le seuil doit être compris entre 0 et 1\."):
        validate_threshold(min_accuracy)


@pytest.mark.parametrize(
    "min_accuracy",
    [0.0, 0.5, 0.90, 1.0],
)
def test_validate_threshold_valid(min_accuracy):
    assert validate_threshold(min_accuracy) is None


def test_save_metrics(tmp_path):
    # Arrange
    metrics = {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.93,
        "f1_score": 0.935,
    }

    report_path = tmp_path / "reports" / "metrics.json"

    # Act
    save_metrics(metrics, report_path)

    # Assert : verifier que le dossier existe
    assert report_path.parent.is_dir()

    # Assert : verifier que le fichier existe
    assert report_path.is_file()

    # Assert : lire le contenu JSON
    with open(report_path, "r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    # Assert : comparer les metriques
    assert saved_metrics == metrics


def test_save_metrics_overwrite(tmp_path):
    # Arrange
    report_path = tmp_path / "metrics.json"

    old_metrics = {"accuracy": 0.85}
    new_metrics = {"accuracy": 0.97}

    # Act : sauvegarder les anciennes metriques
    save_metrics(old_metrics, report_path)

    # Act : sauvegarder les nouvelles metriques
    save_metrics(new_metrics, report_path)

    # Assert : lire le rapport final
    with open(report_path, "r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    # Assert : verifier que seules les nouvelles
    # metriques sont presentes
    assert saved_metrics == new_metrics


@pytest.fixture
def trained_model_path(tmp_path):
    model_path = tmp_path / "models" / "iris_model.joblib"

    train_model(model_path)

    return model_path


def test_train_model(trained_model_path):
    # Verifier que le modele a ete sauvegarde
    assert trained_model_path.is_file()

    # Verifier que le fichier n'est pas vide
    assert trained_model_path.stat().st_size > 0


def test_evaluate_model(trained_model_path):
    # Act
    metrics = evaluate_model(trained_model_path)

    # Assert : verifier le type
    assert isinstance(metrics, dict)

    # Assert : verifier les cles du dictionnaire
    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert set(metrics) == expected_keys

    # Assert : verifier les bornes des metriques
    for value in metrics.values():
        assert 0 <= value <= 1

    # Assert : verifier la performance du modele
    assert metrics["accuracy"] >= 0.90


def test_main_success(monkeypatch):
    # Arrange : configurer le seuil
    monkeypatch.setenv("MIN_ACCURACY", "0.90")

    # Arrange : simuler les metriques
    monkeypatch.setattr(
        evaluation,
        "evaluate_model",
        lambda: {"accuracy": 0.95},
    )

    # Arrange : neutraliser la sauvegarde
    monkeypatch.setattr(
        evaluation,
        "save_metrics",
        lambda metrics: None,
    )

    # Act
    result = evaluation.main()

    # Assert
    assert result == 0


def test_main_failure(monkeypatch):
    # Arrange : configurer le seuil
    monkeypatch.setenv("MIN_ACCURACY", "0.90")

    # Arrange : simuler un mauvais modele
    monkeypatch.setattr(evaluation, "evaluate_model", lambda: {"accuracy": 0.70})

    # Arrange : neutraliser la sauvegarde
    monkeypatch.setattr(evaluation, "save_metrics", lambda metrics: None)

    # Act
    result = evaluation.main()

    # Assert
    assert result == 1


@pytest.mark.parametrize(
    "accuracy, expected",
    [
        (0.89, 1),
        (0.90, 0),
        (0.91, 0),
    ],
)
def test_main_default_threshold(
    monkeypatch,
    accuracy,
    expected,
):
    # Arrange : supprimer la variable d'environnement
    monkeypatch.delenv("MIN_ACCURACY", raising=False)

    # Arrange : simuler les metriques
    monkeypatch.setattr(
        evaluation,
        "evaluate_model",
        lambda: {"accuracy": accuracy},
    )

    # Arrange : neutraliser la sauvegarde
    monkeypatch.setattr(evaluation, "save_metrics", lambda metrics: None)

    # Act
    result = evaluation.main()

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        "1.5",
        "-0.1",
        "nan",
        "abc",
        "",
    ],
)
def test_main_invalid_threshold(
    monkeypatch,
    invalid_threshold,
):
    # Arrange
    monkeypatch.setenv("MIN_ACCURACY", invalid_threshold)

    # Act et Assert
    with pytest.raises(ValueError):
        evaluation.main()


def test_main_saves_metrics_on_failure(monkeypatch):
    # Arrange
    metrics = {"accuracy": 0.50}
    saved_metrics = []

    def fake_save_metrics(metrics):
        saved_metrics.append(metrics)

    monkeypatch.setenv("MIN_ACCURACY", "0.90")

    monkeypatch.setattr(
        evaluation,
        "evaluate_model",
        lambda: metrics,
    )

    monkeypatch.setattr(
        evaluation,
        "save_metrics",
        fake_save_metrics,
    )

    # Act
    result = evaluation.main()

    # Assert : le modele est rejete
    assert result == 1

    # Assert : les metriques ont ete sauvegardees
    assert saved_metrics == [metrics]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_cli_success(
    tmp_path,
    trained_model_path,
):
    # Arrange
    env = os.environ.copy()

    env["MIN_ACCURACY"] = "0.90"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # Act
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Assert : verifier le code de sortie
    assert result.returncode == 0

    # Assert : verifier le message de validation
    assert "Modele valide" in result.stdout

    # Assert : verifier la creation du rapport
    report_path = tmp_path / "reports" / "metrics.json"

    assert report_path.is_file()


@pytest.fixture
def poor_model_path(tmp_path):
    X_train, _, y_train, _ = load_data()

    model = DummyClassifier(
        strategy="most_frequent",
    )

    model.fit(X_train, y_train)

    model_path = tmp_path / "models" / "iris_model.joblib"

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, model_path)

    return model_path


def test_evaluate_cli_failure(
    tmp_path,
    poor_model_path,
):
    # Arrange
    env = os.environ.copy()

    env["MIN_ACCURACY"] = "0.90"
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    # Act
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Assert : verifier le code de sortie
    assert result.returncode == 1

    # Assert : verifier le message de rejet
    assert "Modele rejete" in result.stdout

    # Assert : verifier la conservation du rapport
    report_path = tmp_path / "reports" / "metrics.json"

    assert report_path.is_file()

    # Assert : lire les metriques
    with open(report_path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Assert : verifier que le seuil n'est pas atteint
    assert metrics["accuracy"] < 0.9
