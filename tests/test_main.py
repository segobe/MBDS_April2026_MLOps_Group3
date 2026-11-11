"""Tests for src/main.py: the split, the quality gate and the full pipeline run.

Synthetic data in tmp_path only, never the real data/, models/ or reports/ folders.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import main as pipeline


def _make_raw_df(days: int) -> pd.DataFrame:
    """Build hourly rows in the hour.csv layout from 2011-01-01 with a daily pattern plus fixed noise.

    Inputs: days, number of full days to generate.
    Outputs: DataFrame with hour.csv columns; cnt = casual + registered, whole numbers.
    Why: the noise keeps same hour last week from being a perfect forecast (its weighted_mae must not be 0).
    """
    ts = pd.date_range("2011-01-01", periods=days * 24, freq="h")
    noise = np.random.default_rng(0).integers(0, 30, len(ts))
    cnt = 20 + 10 * ts.hour.to_numpy() + noise
    return pd.DataFrame(
        {
            "instant": np.arange(len(ts)) + 1, "dteday": ts.strftime("%Y-%m-%d"), "season": 1, "yr": 0,
            "mnth": ts.month, "hr": ts.hour, "holiday": 0, "weekday": (ts.day_of_week + 1) % 7,
            "workingday": (ts.day_of_week < 5).astype(int), "weathersit": 1, "temp": 0.5, "atemp": 0.5,
            "hum": 0.5, "windspeed": 0.1, "casual": cnt // 4, "registered": cnt - cnt // 4, "cnt": cnt,
        }
    )


def _make_inference_df() -> pd.DataFrame:
    """Build the agreed inference file: 9 days of history up to 16:00 on D - 1, plus day D with empty demand.

    Inputs: none.
    Outputs: DataFrame in the hour.csv layout; the last 24 rows (2011-03-11) have no demand.
    Why: mirrors the 18:00 export from TEAM_BRIEF (inference input), including the missing evening hours.
    """
    df = _make_raw_df(70).iloc[60 * 24:].reset_index(drop=True)
    df.loc[df.index >= 9 * 24, ["casual", "registered", "cnt"]] = np.nan
    evening_before = (df["dteday"] == "2011-03-10") & (df["hr"] >= 17)
    return df[~evening_before]


def _patch_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, inference_df: pd.DataFrame) -> None:
    """Point every SETTINGS path at tmp_path, write the input files and make the model tiny.

    Inputs: monkeypatch, pytest's patcher; tmp_path, the temporary folder; inference_df, the inference file.
    Outputs: none; SETTINGS is restored automatically after the test.
    Why: main() runs exactly as in production, but fast and without touching the real folders.
    """
    files = {
        "raw_data": "raw/hour.csv", "processed_data": "processed/clean.csv", "model": "models/model.joblib",
        "inference_data": "inference/inference_input.csv", "predictions": "reports/predictions.csv",
    }
    for key, name in files.items():
        monkeypatch.setitem(pipeline.SETTINGS["paths"], key, str(tmp_path / name))
    monkeypatch.setitem(pipeline.SETTINGS["model"], "max_iter", 10)
    for key, df in [("raw_data", _make_raw_df(60)), ("inference_data", inference_df)]:
        (tmp_path / files[key]).parent.mkdir(parents=True)
        df.to_csv(tmp_path / files[key], index=False)


def test_unit_split_time_ordered_sizes_and_order() -> None:
    """100 rows with test 0.2 and val 0.2 give 60 / 20 / 20, no shared rows, train before val before test."""
    X, y = pd.DataFrame({"t": range(100)}), pd.Series(range(100))
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(X, y, 0.2, 0.2, 42, False, True)
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert X_train["t"].max() < X_val["t"].min() and X_val["t"].max() < X_test["t"].min()
    assert y_test.tolist() == list(range(80, 100))


@pytest.mark.parametrize("stratify", [False, True])
def test_unit_split_random_path(stratify: bool) -> None:
    """The random path (course contract) gives 60 / 20 / 20 rows covering every row once, with or without stratify."""
    X, y = pd.DataFrame({"t": range(100)}), pd.Series([0, 1] * 50)
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.split_data(X, y, 0.2, 0.2, 42, stratify, False)
    assert (len(X_train), len(X_val), len(X_test)) == (60, 20, 20)
    assert sorted(pd.concat([X_train, X_val, X_test])["t"]) == list(range(100))


@pytest.mark.parametrize(
    "rows_x, rows_y, test_size, val_size, stratify, match",
    [
        (100, 100, 0.0, 0.2, False, r"SETTINGS\['split'\]"),
        (100, 100, 0.5, 0.5, False, r"SETTINGS\['split'\]"),
        (100, 100, 0.2, 0.2, True, r"stratify.*time_ordered"),
        (100, 99, 0.2, 0.2, False, r"100.*99"),
        (2, 2, 0.4, 0.4, False, r"2 rows.*no training rows"),
    ],
)
def test_unit_split_errors(
    rows_x: int, rows_y: int, test_size: float, val_size: float, stratify: bool, match: str
) -> None:
    """Bad sizes, stratify with time order, unequal lengths and an empty train part raise ValueError."""
    X, y = pd.DataFrame({"t": range(rows_x)}), pd.Series(range(rows_y))
    with pytest.raises(ValueError, match=match):
        pipeline.split_data(X, y, test_size, val_size, 42, stratify, True)


@pytest.mark.parametrize(
    "value, comparison, expected",
    [(0.8, "<=", True), (0.9, "<=", False), (0.9, ">=", True), (0.8, ">=", False)],
)
def test_unit_gate_met_both_directions(value: float, comparison: str, expected: bool) -> None:
    """The gate compares the metric with the threshold 0.85 in the configured direction."""
    gate = {"metric_key": "weighted_mae_ratio", "comparison": comparison, "threshold": 0.85}
    assert pipeline._gate_met({"weighted_mae_ratio": value}, gate) is expected


@pytest.mark.parametrize(
    "gate, match",
    [
        ({"metric_key": "weighted_mae_ratio", "comparison": "<", "threshold": 0.85}, r"comparison"),
        ({"metric_key": "rmse", "comparison": "<=", "threshold": 0.85}, r"rmse.*weighted_mae_ratio"),
    ],
)
def test_unit_gate_errors(gate: dict, match: str) -> None:
    """An unknown comparison or a missing metric key raises ValueError naming what to check."""
    with pytest.raises(ValueError, match=match):
        pipeline._gate_met({"weighted_mae_ratio": 0.8}, gate)


def test_unit_resolve_paths_under_project_root(tmp_path: Path) -> None:
    """A relative path is placed under PROJECT_ROOT; an absolute path is kept."""
    resolved = pipeline._resolve_paths({"raw_data": "data/raw/hour.csv", "model": str(tmp_path / "model.joblib")})
    assert resolved["raw_data"] == pipeline.PROJECT_ROOT / "data" / "raw" / "hour.csv"
    assert resolved["model"] == tmp_path / "model.joblib"


@pytest.mark.parametrize("threshold, status", [(100.0, "Quality gate met"), (0.01, "Quality gate NOT met")])
def test_integration_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture, threshold: float, status: str
) -> None:
    """main() writes all three artifacts, the model reloads, 24 forecasts have ts first, the gate status is logged."""
    from src.utils import load_model

    _patch_settings(monkeypatch, tmp_path, _make_inference_df())
    monkeypatch.setitem(pipeline.SETTINGS["quality_gate"], "threshold", threshold)
    with caplog.at_level(logging.INFO):
        pipeline.main()
    predictions = pd.read_csv(tmp_path / "reports" / "predictions.csv", parse_dates=["ts"])
    assert (tmp_path / "processed" / "clean.csv").exists()
    assert hasattr(load_model(tmp_path / "models" / "model.joblib"), "predict")
    assert list(predictions.columns) == ["ts", "prediction"] and len(predictions) == 24
    assert (predictions["ts"].dt.strftime("%Y-%m-%d") == "2011-03-11").all()
    assert status in caplog.text


def test_integration_missing_feature_column_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A feature column in SETTINGS that the data does not have stops main() before training."""
    _patch_settings(monkeypatch, tmp_path, _make_inference_df())
    monkeypatch.setitem(pipeline.SETTINGS["features"], "calendar_columns", ["hr", "temperature"])
    with pytest.raises(ValueError, match=r"temperature.*SETTINGS\['features'\]"):
        pipeline.main()


def test_integration_missing_tomorrow_rows_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An inference file without rows of missing demand raises ValueError naming the inference file setting."""
    _patch_settings(monkeypatch, tmp_path, _make_raw_df(10))
    with pytest.raises(ValueError, match=r"SETTINGS\['paths'\]\['inference_data'\]"):
        pipeline.main()
