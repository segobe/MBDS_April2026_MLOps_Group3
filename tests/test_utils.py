"""Tests for src/utils.py: shared CSV and model read/write helpers.

Every test uses tmp_path, never the real data/, models/ or reports/ folders.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src import utils


def _small_frame() -> pd.DataFrame:
    """Build a 5-row DataFrame with an integer and a timestamp-like text column.

    Inputs: none.
    Outputs: DataFrame with columns cnt (int) and ts (str).
    Why: one shared fixture keeps the CSV tests short and identical.
    """
    return pd.DataFrame(
        {
            "cnt": [16, 40, 32, 13, 1],
            "ts": [
                "2012-01-01 00:00:00",
                "2012-01-01 01:00:00",
                "2012-01-01 02:00:00",
                "2012-01-01 03:00:00",
                "2012-01-01 04:00:00",
            ],
        }
    )


def _fitted_model() -> LinearRegression:
    """Fit a LinearRegression on 10 rows that follow y = 2x + 1.

    Inputs: none.
    Outputs: a fitted LinearRegression.
    Why: a tiny model with a known formula lets tests check exact predictions.
    """
    x = np.arange(10).reshape(-1, 1)
    y = 2 * x.ravel() + 1
    return LinearRegression().fit(x, y)


def test_unit_save_and_load_csv_roundtrip(tmp_path: Path) -> None:
    """A DataFrame saved with save_csv and read with load_csv comes back unchanged."""
    original = _small_frame()
    filepath = tmp_path / "x.csv"

    utils.save_csv(original, filepath)
    loaded = utils.load_csv(filepath)

    pd.testing.assert_frame_equal(loaded, original)


def test_unit_load_csv_missing_file_raises(tmp_path: Path) -> None:
    """load_csv raises FileNotFoundError naming the path when the file is missing."""
    filepath = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match=re.escape(str(filepath))):
        utils.load_csv(filepath)


def test_unit_save_csv_creates_parent_folders(tmp_path: Path) -> None:
    """save_csv creates missing parent folders and writes no index column."""
    filepath = tmp_path / "a" / "b" / "x.csv"

    utils.save_csv(_small_frame(), filepath)

    assert filepath.exists()
    assert list(pd.read_csv(filepath).columns) == ["cnt", "ts"]


@pytest.mark.parametrize("bad_input", [[1, 2, 3], {"cnt": [1]}, "not a frame"])
def test_unit_save_csv_rejects_non_dataframe(tmp_path: Path, bad_input: object) -> None:
    """save_csv raises TypeError naming the received type for anything but a DataFrame."""
    with pytest.raises(TypeError, match=type(bad_input).__name__):
        utils.save_csv(bad_input, tmp_path / "x.csv")


def test_unit_save_model_creates_parent_folders(tmp_path: Path) -> None:
    """save_model creates missing parent folders and writes the model file."""
    filepath = tmp_path / "models" / "nested" / "model.joblib"

    utils.save_model(_fitted_model(), filepath)

    assert filepath.exists()


def test_unit_load_model_missing_file_raises(tmp_path: Path) -> None:
    """load_model raises FileNotFoundError naming the path and how to create the file."""
    filepath = tmp_path / "missing.joblib"

    with pytest.raises(FileNotFoundError, match=re.escape(str(filepath))):
        utils.load_model(filepath)
    with pytest.raises(FileNotFoundError, match=re.escape("python -m src.main")):
        utils.load_model(filepath)


def test_unit_accepts_string_paths(tmp_path: Path) -> None:
    """save_csv and load_csv accept a plain string path as well as a Path."""
    filepath = str(tmp_path / "x.csv")

    utils.save_csv(_small_frame(), filepath)
    loaded = utils.load_csv(filepath)

    pd.testing.assert_frame_equal(loaded, _small_frame())


def test_integration_save_and_load_model_same_predictions(tmp_path: Path) -> None:
    """A reloaded model gives identical predictions, including a hand-computed value."""
    model = _fitted_model()
    filepath = tmp_path / "model.joblib"
    x_new = np.array([[3], [20]])

    utils.save_model(model, filepath)
    reloaded = utils.load_model(filepath)

    np.testing.assert_array_equal(reloaded.predict(x_new), model.predict(x_new))
    assert reloaded.predict(np.array([[20]]))[0] == pytest.approx(41.0)
