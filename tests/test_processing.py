import pytest

from src.preprocessing import calculate_mean, normalize_values, remove_missing_values


@pytest.mark.parametrize(
    "values, expected",
    [
        ([10, 20, 30], [0.0, 0.5, 1.0]),
        ([-10, 0, 10], [0.0, 0.5, 1.0]),
        ([5, 5, 5], [0.0, 0.0, 0.0]),
        ([], []),
        ([42], [0.0]),
        ([1.5, 2.5, 3.5], [0.0, 0.5, 1.0]),
    ],
)
def test_normalize_values(values, expected):
    assert normalize_values(values) == expected


@pytest.fixture
def data_with_missing_values():
    return [10, None, 20, None, 30]


def test_remove_missing_values(data_with_missing_values):
    result = remove_missing_values(data_with_missing_values)
    assert result == [10, 20, 30]


def test_remove_missing_values_empty():
    assert remove_missing_values([]) == []


def test_remove_missing_values_all_none():
    assert remove_missing_values([None, None]) == []


@pytest.mark.parametrize("values, expected", [([1, 2, 3], 2), ([5], 5), ([-10, 10], 0)])
def test_calculate_mean(values, expected):
    assert calculate_mean(values) == expected


def test_calculate_mean_empty():
    with pytest.raises(ValueError):
        calculate_mean([])
