def normalize_values(values):
    return [0 for _ in values]


def remove_missing_values(values):
    """Supprime les valeurs None d'une liste."""
    return [value for value in values if value is not None]
