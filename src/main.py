"""Run the bike demand pipeline end to end with one command: python -m src.main.

Serves README "1. Business case": Solution (nightly batch, a forecast for every hour of tomorrow)
and Goal and success metrics (quality gate: weighted_mae_ratio <= 0.85 on the validation split).
Orchestrates only: the logic here is the split, the gate check and the selection of tomorrow's rows;
every setting lives in SETTINGS below (moved to config.yaml in Session 10, same keys).
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import load_csv, load_model, save_csv, save_model
from src.validate import validate_dataframe

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SETTINGS = {
    "paths": {
        "raw_data": "data/raw/hour.csv",
        "processed_data": "data/processed/clean.csv",
        "model": "models/model.joblib",
        "inference_data": "data/inference/inference_input.csv",
        "predictions": "reports/predictions.csv",
    },
    "problem": {"target_column": "cnt", "problem_type": "regression", "id_column": "ts"},
    "split": {"test_size": 0.126, "val_size": 0.126, "random_state": 42, "stratify": False, "time_ordered": True},
    "cleaning": {"cutoff_hour": 17, "demand_column": "cnt"},
    "features": {
        "calendar_columns": ["hr", "weekday", "mnth", "season", "holiday", "workingday"],
        "past_demand_columns": [
            "lag_48", "lag_168", "mean_168_to_cutoff", "lag_24_before_cutoff", "last_hour_before_cutoff",
        ],
    },
    "model": {
        "loss": "quantile", "quantile": 2 / 3, "learning_rate": 0.05, "max_iter": 300,
        "max_leaf_nodes": 31, "min_samples_leaf": 40, "early_stopping": False,
    },
    "evaluation": {"under_weight": 2.0, "baseline_column": "lag_168"},
    "validation": {
        "required_columns": [
            "dteday", "hr", "weekday", "mnth", "season", "holiday", "workingday",
            "cnt", "casual", "registered", "ts", "is_imputed",
            "lag_48", "lag_168", "mean_168_to_cutoff", "lag_24_before_cutoff", "last_hour_before_cutoff",
        ],
        "id_column": "ts",
        "value_ranges": {"hr": [0, 23], "cnt": [0, None]},
        "allowed_values": {"holiday": [0, 1], "workingday": [0, 1]},
        "integer_columns": ["cnt"],
        "sum_rules": {"cnt": ["casual", "registered"]},
    },
    "quality_gate": {"metric_key": "weighted_mae_ratio", "comparison": "<=", "threshold": 0.85},
}


def _check_split_inputs(
    X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float, stratify: bool, time_ordered: bool
) -> None:
    """Stop when the split settings or the inputs cannot give three valid parts.

    Inputs: X and y, features and target; test_size and val_size, shares of all rows;
        stratify and time_ordered, the split mode flags.
    Outputs: none.
    Raises: ValueError for sizes outside (0, 1) or summing to 1 or more, both flags set, or unequal lengths.
    Why: a wrong split silently leaks the future or scores the model on too few rows.
    """
    if not (0 < test_size < 1 and 0 < val_size < 1) or test_size + val_size >= 1:
        raise ValueError(
            f"test_size {test_size} and val_size {val_size} must each be between 0 and 1 and sum to less than 1. "
            "Check SETTINGS['split']."
        )
    if stratify and time_ordered:
        raise ValueError(
            "stratify and time_ordered cannot both be true. "
            "Check SETTINGS['split']['stratify'] and SETTINGS['split']['time_ordered']."
        )
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}. Pass features and target of the same rows.")


def _split_time_ordered(X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float) -> tuple:
    """Split by row order: the last rows are test, the rows before them validation.

    Inputs: X and y, sorted by time; test_size and val_size, shares of all rows.
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test).
    Raises: ValueError if the train part would be empty.
    Why: the model is always scored on hours after the ones it learned from (notebook section 4).
    """
    n_test, n_val = round(test_size * len(X)), round(val_size * len(X))
    n_train = len(X) - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Splitting {len(X)} rows into validation {n_val} and test {n_test} leaves no training rows. "
            "Check SETTINGS['split'] and the size of the data."
        )
    cuts = [slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, len(X))]
    return tuple(X.iloc[cut] for cut in cuts) + tuple(y.iloc[cut] for cut in cuts)


def _split_random(
    X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float, random_state: int, stratify: bool
) -> tuple:
    """Split with train_test_split twice, optionally stratified by y (course contract).

    Inputs: X and y; test_size and val_size, shares of all rows; random_state, the seed; stratify, keep y shares.
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test).
    Why: kept for the course contract; this project uses time_ordered=True.
    """
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_size / (1 - test_size), random_state=random_state,
        stratify=y_rest if stratify else None,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float, val_size: float,
    random_state: int, stratify: bool, time_ordered: bool = False,
) -> tuple:
    """Split features and target into train, validation and test parts.

    Inputs: X and y; test_size and val_size, shares of all rows; random_state and stratify, for the
        random path; time_ordered, keep row order (SETTINGS passes True).
    Outputs: (X_train, X_val, X_test, y_train, y_val, y_test).
    Raises: ValueError from _check_split_inputs or _split_time_ordered.
    Why: the split happens before anything is fitted, so no information leaks from validation or test.
    """
    _check_split_inputs(X, y, test_size, val_size, stratify, time_ordered)
    if time_ordered:
        parts = _split_time_ordered(X, y, test_size, val_size)
    else:
        parts = _split_random(X, y, test_size, val_size, random_state, stratify)
    logger.info("Split %d rows: train %d, validation %d, test %d", len(X), len(parts[0]), len(parts[1]), len(parts[2]))
    return parts


def _resolve_paths(paths: dict) -> dict[str, Path]:
    """Turn every path in SETTINGS['paths'] into an absolute path under PROJECT_ROOT.

    Inputs: paths, the SETTINGS['paths'] dictionary (relative or absolute strings).
    Outputs: the same keys with absolute Path values; absolute inputs stay as they are.
    Why: python -m src.main then works from any folder, not only the repo root.
    """
    return {name: PROJECT_ROOT / path for name, path in paths.items()}


def _gate_met(metrics: dict[str, float], gate: dict) -> bool:
    """Check one metric against the quality gate.

    Inputs: metrics, evaluate_model output; gate, SETTINGS['quality_gate'] (metric_key, comparison, threshold).
    Outputs: True if the metric meets the threshold in the given direction.
    Raises: ValueError for an unknown comparison or a metric key that is not in metrics.
    Why: the release decision is a setting, not code (README, Goal and success metrics).
    """
    if gate["comparison"] not in ("<=", ">="):
        raise ValueError(
            f"Unknown comparison '{gate['comparison']}', use '<=' or '>='. "
            "Check SETTINGS['quality_gate']['comparison']."
        )
    if gate["metric_key"] not in metrics:
        raise ValueError(
            f"Metric '{gate['metric_key']}' is not in the evaluation output {sorted(metrics)}. "
            "Check SETTINGS['quality_gate']['metric_key'] and SETTINGS['evaluation']['baseline_column']."
        )
    value = metrics[gate["metric_key"]]
    return value <= gate["threshold"] if gate["comparison"] == "<=" else value >= gate["threshold"]


def _log_split_ranges(ts: pd.Series, parts: dict[str, pd.DataFrame]) -> None:
    """Log the first and last timestamp of each split part.

    Inputs: ts, the timestamps of all cleaned rows; parts, {name: X part} sharing ts's index.
    Outputs: none.
    Why: the log shows that train ends before validation starts, as evidence for the chronological split.
    """
    # The warm-up rows dropped by clean_data move the boundaries about a day off Q3/Q4 2012 (accepted in the brief).
    for name, part in parts.items():
        logger.info("%s split: %s to %s", name, ts.loc[part.index].min(), ts.loc[part.index].max())


def _check_feature_columns(X_train: pd.DataFrame) -> None:
    """Stop when a feature column from SETTINGS is not in the training data.

    Inputs: X_train, the training features.
    Outputs: none.
    Raises: ValueError listing the missing columns.
    Why: fails before the fit with a clear message instead of a scikit-learn error mid-pipeline.
    """
    features = SETTINGS["features"]
    missing = [c for c in features["calendar_columns"] + features["past_demand_columns"] if c not in X_train.columns]
    if missing:
        raise ValueError(f"Feature columns {missing} are not in the training data. Check SETTINGS['features'].")


def _features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target from the features; ts is dropped from the features.

    Inputs: df, cleaned rows.
    Outputs: (X, y) with X = every column except the target and the id column.
    Why: training and inference must pass the model exactly the same columns.
    """
    problem = SETTINGS["problem"]
    return df.drop(columns=[problem["target_column"], problem["id_column"]]), df[problem["target_column"]]


def _report_gate(metrics: dict[str, float]) -> None:
    """Log whether the validation metrics meet the quality gate; never stops the run.

    Inputs: metrics, evaluate_model output on the validation split.
    Outputs: none.
    Why: the logged gate status is the evidence the README cites; tuning happens in a notebook, not here.
    """
    gate = SETTINGS["quality_gate"]
    met = _gate_met(metrics, gate)
    args = (gate["metric_key"], metrics[gate["metric_key"]], gate["comparison"], gate["threshold"])
    if met:
        logger.info("Quality gate met: %s = %.3f (%s %.2f)", *args)
    else:
        logger.warning("Quality gate NOT met: %s = %.3f (needs %s %.2f)", *args)


def _run_training(paths: dict[str, Path]) -> None:
    """Steps 3 to 8: load, clean, validate, split, train, save and check the gate on validation.

    Inputs: paths, the resolved SETTINGS['paths'].
    Outputs: none; writes the processed data and the model.
    Why: every training run starts from the untouched raw file and is scored on unseen hours.
    """
    problem, target = SETTINGS["problem"], SETTINGS["problem"]["target_column"]
    clean = clean_dataframe(load_raw_data(paths["raw_data"]), target_column=target, **SETTINGS["cleaning"])
    validate_dataframe(clean, target_column=target, **SETTINGS["validation"])
    save_csv(clean, paths["processed_data"])
    X, y = _features_and_target(clean)
    X_train, X_val, X_test, y_train, y_val, _ = split_data(X, y, **SETTINGS["split"])
    _log_split_ranges(clean[problem["id_column"]], {"train": X_train, "validation": X_val, "test": X_test})
    _check_feature_columns(X_train)
    model = train_model(
        X_train, y_train, get_feature_preprocessor(**SETTINGS["features"]), problem["problem_type"],
        SETTINGS["model"], SETTINGS["split"]["random_state"],
    )
    save_model(model, paths["model"])
    _report_gate(evaluate_model(model, X_val, y_val, problem["problem_type"], **SETTINGS["evaluation"]))


def _run_inference(paths: dict[str, Path]) -> None:
    """Step 9: clean and validate the inference file, forecast tomorrow's rows and save them.

    Inputs: paths, the resolved SETTINGS['paths'].
    Outputs: none; writes the predictions with ts first.
    Raises: ValueError if the inference file has no rows with missing demand.
    Why: the 18:00 batch turns history up to the cutoff into tomorrow's hourly forecast.
    """
    demand = SETTINGS["cleaning"]["demand_column"]
    clean = clean_dataframe(load_csv(paths["inference_data"]), target_column=None, **SETTINGS["cleaning"])
    validate_dataframe(clean, **SETTINGS["validation"])
    # Tomorrow = rows with missing demand. Not is_imputed: clean_data inserts the hours after the
    # cutoff on D - 1 with demand 0, and those must not be forecast.
    tomorrow = clean[clean[demand].isna()]
    if tomorrow.empty:
        raise ValueError(
            f"The inference file has no rows with missing '{demand}' to forecast. "
            "Check SETTINGS['paths']['inference_data']."
        )
    predictions = run_inference(load_model(paths["model"]), _features_and_target(tomorrow)[0])
    predictions.insert(0, "ts", tomorrow[SETTINGS["problem"]["id_column"]])
    save_csv(predictions, paths["predictions"])


def main() -> None:
    """Run the whole pipeline from SETTINGS: train, check the gate, forecast tomorrow.

    Inputs: none; every value comes from SETTINGS.
    Outputs: none; writes the processed data, the model and the predictions.
    Why: one command that teammates, CI and the nightly batch all run the same way.
    """
    # 1. The only logging setup in the project
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # 2. Resolve every path against PROJECT_ROOT
    paths = _resolve_paths(SETTINGS["paths"])
    # 3 to 8. Load, clean, validate, save, split, train, save the model, check the gate
    _run_training(paths)
    # 9. Inference for tomorrow
    _run_inference(paths)
    # 10. Where the artifacts are
    logger.info("Artifacts: %s | %s | %s", paths["processed_data"], paths["model"], paths["predictions"])


if __name__ == "__main__":
    main()
