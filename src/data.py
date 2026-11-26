from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    """Charge et separe les donnees Iris."""
    iris = load_iris()

    return train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
