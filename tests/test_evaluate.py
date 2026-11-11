"""Tests for src/evaluate.py: weighted_mae, the same-hour-last-week ratio and input errors.

Every test uses tiny synthetic data and a constant DummyRegressor, never the real data/ folder.
"""
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.evaluate import evaluate_model


def _make_X() -> pd.DataFrame:
    """Build 2 evaluation rows with a same-hour-last-week column.

    Inputs: none.
    Outputs: DataFrame with hr and lag_168 (baseline predictions 9 and 6).
    Why: two rows are enough to work out every metric by hand.
    """
    return pd.DataFrame({"hr": [8, 9], "lag_168": [9.0, 6.0]})


def _constant_model(value: float, X: pd.DataFrame) -> DummyRegressor:
    """Fit a DummyRegressor that predicts value for every row.

    Inputs: value, the constant prediction; X, the features to fit on.
    Outputs: a fitted DummyRegressor.
    Why: a fixed prediction makes the expected metrics easy to compute by hand.
    """
    model = DummyRegressor(strategy="constant", constant=value)
    model.fit(X, pd.Series([0.0] * len(X)))
    return model


def test_unit_known_weighted_mae() -> None:
    """Hand-computed weighted_mae, mae and under_share come back as plain Python floats."""
    X = _make_X()
    model = _constant_model(8, X)

    metrics = evaluate_model(model, X, pd.Series([10, 6]), "regression")

    assert metrics["weighted_mae"] == pytest.approx(3.0)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["under_share"] == pytest.approx(0.5)
    for key in ("weighted_mae", "mae", "under_share"):
        assert type(metrics[key]) is float, f"{key} is {type(metrics[key]).__name__}"


def test_unit_baseline_ratio() -> None:
    """Same-hour-last-week weighted_mae and the model / baseline ratio match hand values."""
    X = _make_X()
    model = _constant_model(8, X)

    metrics = evaluate_model(
        model, X, pd.Series([10, 6]), "regression", baseline_column="lag_168"
    )

    # baseline 9 vs actual 10: under by 1, weight 2 -> 2; 6 vs 6 -> 0; mean 1.0
    assert metrics["baseline_weighted_mae"] == pytest.approx(1.0)
    assert metrics["weighted_mae_ratio"] == pytest.approx(3.0)
    assert type(metrics["baseline_weighted_mae"]) is float
    assert type(metrics["weighted_mae_ratio"]) is float


def test_unit_negative_predictions_clipped() -> None:
    """Negative predictions are clipped at 0 before scoring (no negative rentals)."""
    X = _make_X()
    model = _constant_model(-5, X)

    metrics = evaluate_model(model, X, pd.Series([0, 0]), "regression")

    # without clipping: |0 - (-5)| = 5 on both rows -> 5.0
    assert metrics["weighted_mae"] == pytest.approx(0.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_unit_no_baseline_keys_without_baseline_column() -> None:
    """Without baseline_column only the three model metrics are returned."""
    X = _make_X()
    model = _constant_model(8, X)

    metrics = evaluate_model(model, X, pd.Series([10, 6]), "regression")

    assert set(metrics) == {"weighted_mae", "mae", "under_share"}


# case: (X_eval, y_eval, extra keyword arguments, text the message must contain)
_ERROR_CASES = {
    "problem_type_not_regression": (
        _make_X(), pd.Series([10, 6]), {"problem_type": "classification"}, "classification",
    ),
    "X_eval_empty": (
        _make_X().iloc[0:0], pd.Series([], dtype=float), {}, r"val_size",
    ),
    "length_mismatch": (
        _make_X(), pd.Series([10, 6, 4]), {}, r"2.*3",
    ),
    "y_eval_missing": (
        _make_X(), pd.Series([10, None]), {}, r"1 missing",
    ),
    "under_weight_not_positive": (
        _make_X(), pd.Series([10, 6]), {"under_weight": 0}, r"under_weight",
    ),
    "baseline_column_absent": (
        _make_X(), pd.Series([10, 6]), {"baseline_column": "lag_999"}, r"lag_999.*baseline_column",
    ),
    "baseline_column_has_missing": (
        _make_X().assign(lag_168=[9.0, None]), pd.Series([10, 6]),
        {"baseline_column": "lag_168"}, r"lag_168.*baseline_column",
    ),
    "baseline_weighted_mae_zero": (
        _make_X(), pd.Series([9, 6]), {"baseline_column": "lag_168"}, r"ratio",
    ),
}


@pytest.mark.parametrize(
    ("X_eval", "y_eval", "kwargs", "match"),
    list(_ERROR_CASES.values()),
    ids=list(_ERROR_CASES),
)
def test_unit_invalid_inputs_raise(
    X_eval: pd.DataFrame, y_eval: pd.Series, kwargs: dict, match: str
) -> None:
    """Each error rule raises ValueError with a message naming what to check."""
    model = _constant_model(8, _make_X())
    call_kwargs = {"problem_type": "regression", **kwargs}

    with pytest.raises(ValueError, match=match):
        evaluate_model(model, X_eval, y_eval, **call_kwargs)


def test_integration_scores_trained_pipeline() -> None:
    """A pipeline from train_model is scored and every metric is a plain float."""
    from src.features import get_feature_preprocessor
    from src.train import train_model

    n_rows = 60
    X = pd.DataFrame({
        "hr": [i % 24 for i in range(n_rows)],
        "weekday": [i % 7 for i in range(n_rows)],
        "lag_168": [float(10 + i % 24) for i in range(n_rows)],
    })
    y = pd.Series([float(12 + i % 24) for i in range(n_rows)])
    preprocessor = get_feature_preprocessor(["hr", "weekday"], ["lag_168"])
    model = train_model(X, y, preprocessor, "regression")

    metrics = evaluate_model(model, X, y, "regression", baseline_column="lag_168")

    assert set(metrics) == {
        "weighted_mae", "mae", "under_share", "baseline_weighted_mae", "weighted_mae_ratio",
    }
    assert all(type(value) is float for value in metrics.values())
