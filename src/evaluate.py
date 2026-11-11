"""
evaluate.py  [Step 6: EVALUATE]
===============================
Purpose   Score the fitted pipeline with weighted_mae and compare it with the "same hour last week" rule
Serves    README, Goal and success metrics: main releases the model only if weighted_mae_ratio <= 0.85
Input     Fitted pipeline + X_eval, y_eval (validation split; test split once, at the end)
Output    Dictionary of plain floats; main makes the go/no-go decision, not this module
Tests     tests/test_evaluate.py
"""
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _check_eval_inputs(
    X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str, under_weight: float
) -> None:
    """Stop when the evaluation inputs or settings cannot give a meaningful score.

    Inputs: X_eval, y_eval, the split to score; problem_type and under_weight, from SETTINGS.
    Outputs: none.
    Raises: ValueError for a non-regression problem, an empty or mismatched split,
        missing target values or a non-positive under_weight.
    Why: a wrong score would silently pass or block the release at the quality gate.
    """
    if problem_type != "regression":
        raise ValueError(
            f"evaluate_model supports only 'regression', got '{problem_type}'. "
            "Check SETTINGS['problem']['problem_type']."
        )
    if X_eval.empty:
        raise ValueError(
            "X_eval is empty, so the validation split has no rows to score. "
            "Check SETTINGS['split']['val_size']."
        )
    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"X_eval has {len(X_eval)} rows but y_eval has {len(y_eval)}. "
            "Pass the features and target of the same split."
        )
    n_missing = int(y_eval.isna().sum())
    if n_missing > 0:
        raise ValueError(
            f"y_eval has {n_missing} missing value(s). Check that validate ran before evaluate."
        )
    if under_weight <= 0:
        raise ValueError(
            f"under_weight must be greater than 0, got {under_weight}. "
            "Check SETTINGS['evaluation']['under_weight']."
        )


def _weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, under_weight: float) -> float:
    """Mean absolute error where under-forecasts count under_weight times.

    Inputs: y_true, actual rentals; y_pred, forecasts; under_weight, weight when y_pred < y_true.
    Outputs: weighted_mae as a plain float.
    Why: a missing bike (under-forecast) costs more than an idle one (README, Goal and success metrics).
    """
    weights = np.where(y_pred < y_true, under_weight, 1.0)
    return float(np.mean(weights * np.abs(y_true - y_pred)))


def _baseline_metrics(
    X_eval: pd.DataFrame, y_true: np.ndarray, model_wmae: float,
    baseline_column: str, under_weight: float,
) -> dict[str, float]:
    """Score the same-hour-last-week rule and the model / baseline ratio.

    Inputs: X_eval, holding the baseline column; y_true, actual rentals; model_wmae, the model score;
        baseline_column and under_weight, from SETTINGS['evaluation'].
    Outputs: {"baseline_weighted_mae", "weighted_mae_ratio"} as plain floats.
    Raises: ValueError if the column is absent, has missing values or scores exactly 0.
    Why: the quality gate compares the model with the better simple rule on the same rows and weight.
    """
    if baseline_column not in X_eval.columns:
        raise ValueError(
            f"Baseline column '{baseline_column}' is not in X_eval. "
            "Check SETTINGS['evaluation']['baseline_column'] and the feature columns."
        )
    baseline = X_eval[baseline_column]
    n_missing = int(baseline.isna().sum())
    if n_missing > 0:
        raise ValueError(
            f"Baseline column '{baseline_column}' has {n_missing} missing value(s). "
            "Check SETTINGS['evaluation']['baseline_column'] and the lag rules in clean_data."
        )
    baseline_wmae = _weighted_mae(y_true, baseline.to_numpy(dtype=float), under_weight)
    if baseline_wmae == 0:
        raise ValueError(
            "Baseline weighted_mae is 0, so weighted_mae_ratio cannot be computed. "
            "Check that the evaluation split is not a copy of last week's demand."
        )
    return {
        "baseline_weighted_mae": baseline_wmae,
        "weighted_mae_ratio": float(model_wmae / baseline_wmae),
    }


def evaluate_model(
    model: Pipeline, X_eval: pd.DataFrame, y_eval: pd.Series, problem_type: str,
    under_weight: float = 2.0, baseline_column: str | None = None,
) -> dict[str, float]:
    """Score the fitted pipeline on unseen data with the business case metric.

    Inputs: model, fitted pipeline; X_eval, y_eval, the validation (or, once, test) split;
        problem_type, under_weight and baseline_column, from SETTINGS.
    Outputs: weighted_mae, mae and under_share; plus baseline_weighted_mae and
        weighted_mae_ratio when baseline_column is given. All plain floats.
    Raises: ValueError for invalid inputs or settings (see _check_eval_inputs, _baseline_metrics).
    Why: main compares weighted_mae_ratio with the 0.85 quality gate before releasing the model.
    """
    _check_eval_inputs(X_eval, y_eval, problem_type, under_weight)
    y_true = y_eval.to_numpy(dtype=float)
    y_pred = np.clip(model.predict(X_eval), 0, None)  # no negative rentals
    metrics = {
        "weighted_mae": _weighted_mae(y_true, y_pred, under_weight),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "under_share": float(np.mean(y_pred < y_true)),
    }
    if baseline_column is not None:
        metrics.update(
            _baseline_metrics(X_eval, y_true, metrics["weighted_mae"], baseline_column, under_weight)
        )
    logger.info("Scored %d rows: %s", len(y_true), metrics)
    return metrics
