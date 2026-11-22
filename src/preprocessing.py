
def normalize_values(values):
    """Normalise une liste de nombres entre 0 et 1."""
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if minimum == maximum:
        return [0.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def remove_missing_values(values):
    """Supprime les valeurs None d'une liste."""
    return [value for value in values if value is not None]
