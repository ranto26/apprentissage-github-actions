from src.preprocessing import (
    normalize_values,
    remove_missing_values,
)


def test_normalize_values():
    result = normalize_values([10, 20, 30])
    assert result == [0.0, 0.5, 1.0]


def test_normalize_empty_list():
    assert normalize_values([]) == []


def test_normalize_constant_values():
    assert normalize_values([5, 5, 5]) == [0.0, 0.0, 0.0]


def test_remove_missing_values():
    result = remove_missing_values([1, None, 3])
    assert result == [1, 3]


def test_remove_missing_values_empty():
    assert remove_missing_values([]) == []
