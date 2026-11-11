"""Tests for src/validate.py: the data quality gate after clean_data.

Every test uses tiny synthetic DataFrames, never the real data/ folder.
"""
import numpy as np
import pandas as pd
import pytest

from src import validate

_RULES = {
    "required_columns": [
        "dteday", "hr", "weekday", "mnth", "season", "holiday", "workingday",
        "cnt", "casual", "registered", "ts", "is_imputed",
        "lag_48", "lag_168", "mean_168_to_cutoff",
    ],
    "id_column": "ts",
    "value_ranges": {"hr": [0, 23], "cnt": [0, None]},
    "allowed_values": {"holiday": [0, 1], "workingday": [0, 1]},
    "integer_columns": ["cnt"],
    "sum_rules": {"cnt": ["casual", "registered"]},
}

# case: (column, row, new value) applied to the valid frame
_EDITS = {
    "duplicate_id": ("ts", 1, pd.Timestamp("2012-01-08 00:00:00")),
    "target_missing": ("cnt", 0, np.nan),
    "hr_above": ("hr", 0, 24),
    "cnt_below": ("cnt", 0, -1.0),
    "holiday_not_allowed": ("holiday", 0, 2),
    "cnt_not_integer": ("cnt", 0, 2.5),
    "sum_mismatch": ("cnt", 0, 17.0),
}


def _make_valid_df() -> pd.DataFrame:
    """Build 5 cleaned hourly rows that satisfy every validation rule.

    Inputs: none.
    Outputs: DataFrame with all required columns; cnt, casual, registered are floats like clean_data output.
    Why: every test starts from known-good data and breaks exactly one rule.
    """
    return pd.DataFrame(
        {
            "dteday": ["2012-01-08"] * 5,
            "hr": [0, 1, 2, 3, 4],
            "weekday": [0] * 5,
            "mnth": [1] * 5,
            "season": [1] * 5,
            "holiday": [0] * 5,
            "workingday": [0] * 5,
            "cnt": [16.0, 40.0, 32.0, 13.0, 1.0],
            "casual": [3.0, 8.0, 5.0, 3.0, 0.0],
            "registered": [13.0, 32.0, 27.0, 10.0, 1.0],
            "ts": pd.date_range("2012-01-08", periods=5, freq="h"),
            "is_imputed": [0] * 5,
            "lag_48": [14.0, 35.0, 30.0, 12.0, 2.0],
            "lag_168": [17.0, 41.0, 29.0, 15.0, 1.0],
            "mean_168_to_cutoff": [190.5] * 5,
        }
    )


def _broken_frame(case: str) -> pd.DataFrame:
    """Return the valid frame with the one change that case describes.

    Inputs: case, a key of _EDITS, or empty, missing_column, or a case that only changes the rules.
    Outputs: the changed DataFrame.
    Why: lets one parametrized test break every rule with one short setup.
    """
    df = _make_valid_df()
    if case == "empty":
        return df.iloc[0:0]
    if case == "missing_column":
        return df.drop(columns=["hr", "lag_48"])
    if case in _EDITS:
        column, row, value = _EDITS[case]
        df.loc[row, column] = value
    return df


def _validate(df: pd.DataFrame, **overrides: object) -> bool:
    """Run validate_dataframe with the team rules, replacing any rule given in overrides.

    Inputs: df, the frame to check; overrides, rule keyword arguments to change.
    Outputs: the return value of validate_dataframe.
    Why: tests mirror how main passes SETTINGS['validation'], changing only what they test.
    """
    return validate.validate_dataframe(df, **{**_RULES, **overrides})


def test_unit_valid_frame_returns_true() -> None:
    """A frame that meets every rule returns True in training mode."""
    assert _validate(_make_valid_df(), target_column="cnt") is True


def test_unit_only_required_columns_given_passes() -> None:
    """With every optional rule left as None, only the required-column check runs."""
    assert validate.validate_dataframe(_make_valid_df(), required_columns=_RULES["required_columns"]) is True


def test_unit_inference_rows_with_empty_demand_pass() -> None:
    """Rows with missing cnt, casual and registered pass without target_column."""
    df = _make_valid_df().head(2)
    df[["cnt", "casual", "registered"]] = np.nan

    assert _validate(df) is True


def test_unit_sum_rule_skipped_in_inference_mode() -> None:
    """cnt != casual + registered passes without target_column and fails with it."""
    df = _broken_frame("sum_mismatch")

    assert _validate(df) is True
    with pytest.raises(ValueError, match="cnt"):
        _validate(df, target_column="cnt")


@pytest.mark.parametrize(
    ("case", "overrides", "match"),
    [
        ("empty", {}, "empty"),
        ("missing_column", {}, r"\['hr', 'lag_48'\]"),
        ("duplicate_id", {}, r"'ts'.*duplicate.*2012-01-08 00:00:00"),
        ("id_absent", {"id_column": "timestamp"}, "timestamp"),
        ("target_missing", {"target_column": "cnt"}, r"'cnt'.*missing.*row 0"),
        ("target_absent", {"target_column": "demand"}, "demand"),
        ("unknown_rule_column", {"value_ranges": {"temp": [0, 1]}}, r"value_ranges.*'temp'"),
        ("hr_above", {}, r"'hr'.*outside \[0, 23\], e\.g\. 24"),
        ("cnt_below", {}, r"'cnt'.*outside \[0, None\], e\.g\. -1"),
        ("holiday_not_allowed", {}, r"'holiday'.*\[0, 1\], e\.g\. 2"),
        ("cnt_not_integer", {}, r"'cnt'.*integer.*e\.g\. 2\.5"),
        ("sum_mismatch", {"target_column": "cnt"}, r"'cnt'.*casual.*registered.*e\.g\. row 0: 17"),
    ],
)
def test_unit_broken_rule_raises_clear_message(case: str, overrides: dict, match: str) -> None:
    """Each broken rule raises ValueError naming the column and an example value."""
    with pytest.raises(ValueError, match=match):
        _validate(_broken_frame(case), **overrides)


@pytest.mark.parametrize(
    ("column", "value", "value_ranges"),
    [("cnt", 10000.0, {"cnt": [0, None]}), ("hr", -3, {"hr": [None, 23]})],
)
def test_unit_open_limit_accepts_values_beyond_it(column: str, value: float, value_ranges: dict) -> None:
    """A None limit means no limit on that side: cnt 10,000 passes with [0, None]."""
    df = _make_valid_df()
    df.loc[0, column] = value

    assert _validate(df, value_ranges=value_ranges) is True


def test_unit_input_unchanged() -> None:
    """validate_dataframe never modifies the caller's DataFrame."""
    df = _make_valid_df()
    before = df.copy()

    _validate(df, target_column="cnt")

    pd.testing.assert_frame_equal(df, before)


def _make_raw_hourly_df(days: int) -> pd.DataFrame:
    """Build a complete raw hourly frame in the hour.csv layout starting 2011-01-01.

    Inputs: days, number of full days to generate.
    Outputs: DataFrame with hour.csv columns; cnt = casual + registered.
    Why: realistic raw input for the clean_data integration test.
    """
    ts = pd.date_range("2011-01-01", periods=days * 24, freq="h")
    registered = np.arange(len(ts)) % 50
    return pd.DataFrame(
        {
            "instant": np.arange(len(ts)) + 1,
            "dteday": ts.strftime("%Y-%m-%d"),
            "season": 1,
            "yr": 0,
            "mnth": ts.month,
            "hr": ts.hour,
            "holiday": 0,
            "weekday": (ts.day_of_week + 1) % 7,
            "workingday": (ts.day_of_week < 5).astype(int),
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.1,
            "casual": 2,
            "registered": registered,
            "cnt": registered + 2,
        }
    )


@pytest.mark.skip(reason="Awaiting clean_data in dev")
def test_integration_clean_output_passes() -> None:
    """clean_dataframe output on 10 synthetic days passes every rule in training mode."""
    from src import clean_data

    cleaned = clean_data.clean_dataframe(_make_raw_hourly_df(10), target_column="cnt")

    assert _validate(cleaned, target_column="cnt") is True
