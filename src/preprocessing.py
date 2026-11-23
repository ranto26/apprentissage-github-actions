def normalize_values(values):
    """Normalise une liste de nombres entre 0 et 1."""
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [0.0 for _ in values]

    return [(value - minimum) / (maximum - minimum) for value in values]


def remove_missing_values(values):
    """Supprime les valeurs None d'une liste."""
    return [value for value in values if value is not None]


def calculate_mean(values):
    """Calcule la moyenne d'une liste de nombres."""
    if not values:
        raise ValueError("La liste ne peut pas être vide")

    return sum(values) / len(values)
