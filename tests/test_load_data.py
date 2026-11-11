"""Tests for src/load_data.py: read the raw hourly file and fail fast on bad input.

Every test uses tmp_path, never the real data/raw/hour.csv.
"""
from pathlib import Path

import pandas as pd
import pytest

from src import load_data

_BAD_CONTENTS = {"empty": "", "header_only": "dteday,hr,cnt\n"}


def _bad_input_path(tmp_path: Path, case: str) -> Path:
    """Create the bad raw-data input named by case and return its path.

    Inputs: tmp_path, the test's temporary folder; case, one of missing, folder, empty, header_only.
    Outputs: path to pass to load_raw_data.
    Why: lets one parametrized test cover all four input rules.
    """
    if case == "missing":
        return tmp_path / "missing.csv"
    if case == "folder":
        return tmp_path
    path = tmp_path / f"{case}.csv"
    path.write_text(_BAD_CONTENTS[case])
    return path


def test_unit_load_small_csv_shape(tmp_path: Path) -> None:
    """A 3-row raw CSV loads with shape (3, 3) and the cnt values as written."""
    filepath = tmp_path / "hour.csv"
    filepath.write_text("dteday,hr,cnt\n2011-01-01,0,16\n2011-01-01,1,40\n2011-01-01,2,32\n")

    df = load_data.load_raw_data(filepath)

    assert df.shape == (3, 3)
    assert df["cnt"].tolist() == [16, 40, 32]


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("missing", FileNotFoundError),
        ("folder", IsADirectoryError),
        ("empty", ValueError),
        ("header_only", ValueError),
    ],
)
def test_unit_input_errors(tmp_path: Path, case: str, expected_error: type) -> None:
    """Each bad input raises its own error type and names SETTINGS['paths']['raw_data']."""
    path = _bad_input_path(tmp_path, case)

    with pytest.raises(expected_error, match="raw_data"):
        load_data.load_raw_data(path)


def test_integration_reads_file_written_by_utils(tmp_path: Path) -> None:
    """A file written by utils.save_csv is read back unchanged, also via a str path."""
    from src import utils

    original = pd.DataFrame({"dteday": ["2011-01-01", "2011-01-01"], "hr": [0, 1], "cnt": [16, 40]})
    filepath = tmp_path / "raw" / "hour.csv"
    utils.save_csv(original, filepath)

    loaded = load_data.load_raw_data(str(filepath))

    pd.testing.assert_frame_equal(loaded, original)
