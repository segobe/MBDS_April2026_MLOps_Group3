"""Tests for src/features.py: the unfitted feature recipe.

Every test uses tiny synthetic DataFrames, never the real data/ folder.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src import features

_CALENDAR = ["hr", "weekday", "mnth", "season", "holiday", "workingday"]
_PAST_DEMAND = ["lag_48", "lag_168", "mean_168_to_cutoff"]


def _make_feature_frame() -> pd.DataFrame:
    """Build 5 rows with the nine feature columns plus three columns that must be dropped.

    Inputs: none.
    Outputs: DataFrame with columns in a different order than the recipe, plus casual, temp, is_imputed.
    Why: proves the recipe keeps only the listed columns, in recipe order.
    """
    return pd.DataFrame(
        {
            "casual": [3, 8, 5, 3, 0],
            "lag_168": [17.0, 41.0, 29.0, 15.0, 1.0],
            "hr": [0, 1, 2, 3, 4],
            "weekday": [0] * 5,
            "temp": [0.24, 0.22, 0.22, 0.24, 0.24],
            "mnth": [1] * 5,
            "season": [1] * 5,
            "holiday": [0] * 5,
            "workingday": [0] * 5,
            "lag_48": [14.0, 35.0, 30.0, 12.0, 2.0],
            "mean_168_to_cutoff": [190.5] * 5,
            "is_imputed": [0] * 5,
        }
    )


def test_unit_returns_unfitted_column_transformer() -> None:
    """get_feature_preprocessor returns a ColumnTransformer that has not been fitted."""
    recipe = features.get_feature_preprocessor(_CALENDAR, _PAST_DEMAND)

    assert isinstance(recipe, ColumnTransformer)
    assert not hasattr(recipe, "transformers_")
    assert recipe.remainder == "drop"


@pytest.mark.parametrize(
    ("calendar", "past_demand", "match"),
    [
        ([], _PAST_DEMAND, r"calendar_columns.*SETTINGS\['features'\]"),
        (_CALENDAR, [], r"past_demand_columns.*SETTINGS\['features'\]"),
        (["hr", "weekday", "hr"], _PAST_DEMAND, r"twice.*'hr'"),
        (_CALENDAR, ["lag_48", "hr"], r"'hr'.*both"),
    ],
)
def test_unit_invalid_column_lists_raise(calendar: list[str], past_demand: list[str], match: str) -> None:
    """An empty list, a duplicate inside a list or an overlap between lists raises ValueError."""
    with pytest.raises(ValueError, match=match):
        features.get_feature_preprocessor(calendar, past_demand)


def test_integration_fit_transform_keeps_nine_columns() -> None:
    """Fitting on data with extra columns keeps exactly the nine features, in recipe order."""
    recipe = features.get_feature_preprocessor(_CALENDAR, _PAST_DEMAND)

    result = recipe.fit_transform(_make_feature_frame())

    assert result.shape == (5, 9)
    assert not np.isnan(result).any()
    np.testing.assert_array_equal(result[0], [0, 0, 1, 1, 0, 0, 14.0, 17.0, 190.5])


def test_integration_missing_feature_column_fails_fast() -> None:
    """Fitting on data without lag_48 stops with scikit-learn's ValueError."""
    recipe = features.get_feature_preprocessor(_CALENDAR, _PAST_DEMAND)

    with pytest.raises(ValueError, match="lag_48"):
        recipe.fit(_make_feature_frame().drop(columns=["lag_48"]))
