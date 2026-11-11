"""Tests for src/infer.py: tomorrow's hourly forecast from the saved model.

Every test uses tiny synthetic DataFrames or tmp_path, never the real data/ or models/ folders.
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src import infer


def _make_infer_frame() -> pd.DataFrame:
    """Build 3 feature rows with index [10, 11, 12].

    Inputs: none.
    Outputs: DataFrame with hr and lag_48, indexed 10 to 12.
    Why: a non-default index proves predictions stay linked to their hour.
    """
    return pd.DataFrame({"hr": [0, 1, 2], "lag_48": [14.0, 35.0, 30.0]}, index=[10, 11, 12])


def _constant_model(value: float) -> DummyRegressor:
    """Fit a DummyRegressor that always predicts value.

    Inputs: value, the constant prediction.
    Outputs: a fitted DummyRegressor.
    Why: a constant model makes every expected prediction known by hand.
    """
    return DummyRegressor(strategy="constant", constant=value).fit(_make_infer_frame(), [0.0, 0.0, 0.0])


def test_unit_prediction_column_same_index_input_unchanged() -> None:
    """Predictions come back in one 'prediction' column, same index, value 7.0, input untouched."""
    X_infer = _make_infer_frame()
    before = X_infer.copy()

    result = infer.run_inference(_constant_model(7), X_infer)

    expected = pd.DataFrame({"prediction": [7.0, 7.0, 7.0]}, index=[10, 11, 12])
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(X_infer, before)


def test_unit_negative_predictions_clipped() -> None:
    """A model predicting -5 gives 0.0, since rentals cannot be negative."""
    result = infer.run_inference(_constant_model(-5), _make_infer_frame())

    assert result["prediction"].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("case", "expected_error", "match"),
    [
        ("empty", ValueError, r"inference.*SETTINGS\['paths'\]\['inference_data'\]"),
        ("include_proba", ValueError, "regression model has no probabilities"),
        ("no_predict", TypeError, r"object.*SETTINGS\['paths'\]\['model'\]"),
    ],
)
def test_unit_invalid_inputs_raise(case: str, expected_error: type, match: str) -> None:
    """Empty input, include_proba=True and a model without predict each raise a clear error."""
    model = object() if case == "no_predict" else _constant_model(7)
    X_infer = _make_infer_frame().iloc[0:0] if case == "empty" else _make_infer_frame()

    with pytest.raises(expected_error, match=match):
        infer.run_inference(model, X_infer, include_proba=case == "include_proba")


def test_integration_saved_model_reloaded_then_infer(tmp_path: Path) -> None:
    """A model saved and reloaded with utils gives the same forecast as before saving."""
    from src import utils

    model = _constant_model(12.5)
    before = infer.run_inference(model, _make_infer_frame())
    utils.save_model(model, tmp_path / "model.joblib")

    after = infer.run_inference(utils.load_model(tmp_path / "model.joblib"), _make_infer_frame())

    pd.testing.assert_frame_equal(after, before)
