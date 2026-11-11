"""Tests for src/train.py: fitting the feature recipe and the model as one Pipeline.

Every test uses about 30 synthetic rows, a passthrough ColumnTransformer and tiny
model parameters, never the real data/, models/ or reports/ folders.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from src.train import train_model

TINY_PARAMS = {
    "loss": "quantile", "quantile": 2 / 3, "max_iter": 10, "min_samples_leaf": 1, "early_stopping": False,
}


def _training_data(n_rows: int = 30) -> tuple[pd.DataFrame, pd.Series]:
    """Build a small feature table and a target that depends on the hour.

    Inputs: n_rows, number of rows to create.
    Outputs: X with columns hr and lag_168, and y named cnt.
    Why: a tiny, deterministic dataset keeps every fit fast and repeatable.
    """
    hr = np.arange(n_rows) % 24
    X = pd.DataFrame({"hr": hr, "lag_168": hr * 10 + 5})
    y = pd.Series(hr * 10 + 7, name="cnt", dtype=float)
    return X, y


def _passthrough_preprocessor() -> ColumnTransformer:
    """Build an unfitted recipe that passes hr and lag_168 through unchanged.

    Inputs: none.
    Outputs: an unfitted ColumnTransformer.
    Why: stands in for features.get_feature_preprocessor so unit tests only test train.
    """
    return ColumnTransformer([("passthrough", "passthrough", ["hr", "lag_168"])])


def _valid_kwargs() -> dict:
    """Collect a full set of valid arguments for train_model.

    Inputs: none.
    Outputs: a dict of keyword arguments that train_model accepts.
    Why: each error test changes exactly one argument of a known-good call.
    """
    X, y = _training_data()
    return {
        "X_train": X,
        "y_train": y,
        "preprocessor": _passthrough_preprocessor(),
        "problem_type": "regression",
        "model_params": dict(TINY_PARAMS),
        "random_state": 42,
    }


def test_unit_returns_fitted_pipeline_one_prediction_per_row() -> None:
    """train_model returns a fitted preprocess-then-model Pipeline with one prediction per row."""
    kwargs = _valid_kwargs()
    pipeline = train_model(**kwargs)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["preprocess", "model"]
    check_is_fitted(pipeline)
    assert pipeline.predict(kwargs["X_train"]).shape == (30,)


def test_unit_model_params_and_seed_reach_the_model() -> None:
    """Values from SETTINGS['model'] and the seed arrive unchanged in the fitted model."""
    model = train_model(**_valid_kwargs()).named_steps["model"]
    assert model.quantile == pytest.approx(2 / 3)
    assert model.max_iter == 10
    assert model.random_state == 42


def test_unit_same_seed_same_predictions() -> None:
    """Two fits with random_state=42 give identical predictions (reproducibility)."""
    first = train_model(**_valid_kwargs())
    second = train_model(**_valid_kwargs())
    X, _ = _training_data()
    np.testing.assert_array_equal(first.predict(X), second.predict(X))


def test_unit_default_params_when_none() -> None:
    """With model_params=None the default model fits and predicts a constant target exactly (7.0)."""
    kwargs = _valid_kwargs()
    kwargs["model_params"] = None
    kwargs["y_train"] = pd.Series([7.0] * 30, name="cnt")
    predictions = train_model(**kwargs).predict(kwargs["X_train"])
    assert predictions == pytest.approx([7.0] * 30)


def test_unit_does_not_modify_caller_data() -> None:
    """train_model leaves the caller's X_train and y_train unchanged."""
    kwargs = _valid_kwargs()
    X_before, y_before = kwargs["X_train"].copy(), kwargs["y_train"].copy()
    train_model(**kwargs)
    pd.testing.assert_frame_equal(kwargs["X_train"], X_before)
    pd.testing.assert_series_equal(kwargs["y_train"], y_before)


def test_unit_logs_rows_columns_and_duration(caplog: pytest.LogCaptureFixture) -> None:
    """train_model logs the training size before and the duration after fitting."""
    with caplog.at_level(logging.INFO, logger="src.train"):
        train_model(**_valid_kwargs())
    assert "Training on 30 rows and 2 columns" in caplog.text
    assert "Training finished in" in caplog.text


_X, _Y = _training_data()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"problem_type": "classification"}, r"classification.*SETTINGS\['problem'\]\['problem_type'\]"),
        ({"X_train": _X.iloc[0:0], "y_train": _Y.iloc[0:0]}, r"empty.*SETTINGS\['split'\]"),
        ({"y_train": _Y.iloc[:-1]}, r"30.*29"),
        ({"y_train": _Y.mask(_Y.index == 3)}, r"1 missing"),
        ({"model_params": {"max_itter": 10}}, r"max_itter.*SETTINGS\['model'\]"),
    ],
    ids=["wrong_problem_type", "empty_train", "length_mismatch", "missing_target", "unknown_param"],
)
def test_unit_invalid_inputs_raise(overrides: dict, match: str) -> None:
    """Each invalid input stops training with a ValueError that names what to check."""
    kwargs = _valid_kwargs()
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        train_model(**kwargs)


def test_integration_features_then_train_then_predict() -> None:
    """The real feature recipe, fitted by train_model, predicts one value per row."""
    from src.features import get_feature_preprocessor

    X, y = _training_data()
    preprocessor = get_feature_preprocessor(calendar_columns=["hr"], past_demand_columns=["lag_168"])
    pipeline = train_model(X, y, preprocessor, "regression", dict(TINY_PARAMS), random_state=42)
    assert pipeline.predict(X).shape == (30,)


def test_integration_saved_pipeline_predicts_the_same(tmp_path: Path) -> None:
    """A pipeline saved and reloaded with utils gives exactly the same predictions."""
    from src.utils import load_model, save_model

    X, _ = _training_data()
    pipeline = train_model(**_valid_kwargs())
    save_model(pipeline, tmp_path / "model.joblib")
    reloaded = load_model(tmp_path / "model.joblib")
    np.testing.assert_array_equal(pipeline.predict(X), reloaded.predict(X))
