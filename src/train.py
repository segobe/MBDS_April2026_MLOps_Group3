"""Fit the feature recipe and the demand model together as one scikit-learn Pipeline.

Serves README section 1, "Goal and success metrics": under-forecasts cost twice as much,
so the model is trained on the 2/3 quantile instead of the mean.
This is the only module in the project that calls .fit(); infer and evaluate only predict.
Model parameters and the seed arrive as parameters from SETTINGS in src/main.py.
"""
import logging
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _check_training_inputs(X_train: pd.DataFrame, y_train: pd.Series, problem_type: str) -> None:
    """Stop before fitting when the problem type or the train split cannot be used.

    Inputs: X_train, the feature rows; y_train, the target values; problem_type, from SETTINGS.
    Outputs: none.
    Raises: ValueError if problem_type is not "regression", X_train is empty,
    X_train and y_train differ in length, or y_train has missing values.
    Why: a clear message here is cheaper than a confusing scikit-learn error mid-fit.
    """
    if problem_type != "regression":
        raise ValueError(
            f"problem_type '{problem_type}' is not supported: train_model only fits regression. "
            "Check SETTINGS['problem']['problem_type']."
        )
    if X_train.empty:
        raise ValueError("The train split is empty (0 rows). Check the split sizes in SETTINGS['split'].")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)} rows. "
            "Check that both come from the same split_data call."
        )
    n_missing = int(y_train.isna().sum())
    if n_missing > 0:
        raise ValueError(
            f"y_train has {n_missing} missing target values. Check clean_data and validate for the target column."
        )


def _build_model(model_params: dict | None, random_state: int) -> HistGradientBoostingRegressor:
    """Create the unfitted model from the SETTINGS parameters and the seed.

    Inputs: model_params, keyword arguments for the model (None means defaults);
    random_state, the seed for reproducible fits.
    Outputs: an unfitted HistGradientBoostingRegressor.
    Raises: ValueError if model_params contains a name the model does not accept.
    Why: a typo in SETTINGS['model'] should name the key to fix, not raise a bare TypeError.
    """
    params = {} if model_params is None else dict(model_params)
    try:
        return HistGradientBoostingRegressor(**params, random_state=random_state)
    except TypeError as err:
        raise ValueError(
            f"Unknown model parameter in {sorted(params)}: {err}. Check the keys in SETTINGS['model']."
        ) from err


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    problem_type: str,
    model_params: dict | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Fit the feature recipe and the model as one Pipeline on the train split.

    Inputs: X_train and y_train, the train split only; preprocessor, the unfitted recipe
    from features; problem_type, from SETTINGS['problem']; model_params, from
    SETTINGS['model'] (None means defaults); random_state, from SETTINGS['split'].
    Outputs: the fitted Pipeline with steps "preprocess" and "model".
    Raises: ValueError for any invalid input (see _check_training_inputs and _build_model).
    Why: one fitted object for preprocessing and model means inference repeats exactly
    the steps that were validated; the logged duration backs the README batch-time estimate.
    """
    _check_training_inputs(X_train, y_train, problem_type)
    model = _build_model(model_params, random_state)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    logger.info("Training on %d rows and %d columns", X_train.shape[0], X_train.shape[1])
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    logger.info("Training finished in %.1f s", time.perf_counter() - start)
    return pipeline
