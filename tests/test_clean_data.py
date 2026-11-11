"""Tests for src/clean_data.py: hourly calendar and cutoff-safe past-demand features.

Synthetic frames only (demand = running hour index), never the real data/ folder.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def _make_hourly_df(days: int) -> pd.DataFrame:
    """Build a complete hourly frame in the hour.csv layout starting 2011-01-01 00:00.

    Inputs: days, number of full days to generate.
    Outputs: DataFrame with hour.csv columns; cnt = registered = running hour index t, casual = 0.
    Why: demand equal to t makes every lag and window mean computable by hand.
    """
    ts = pd.date_range("2011-01-01", periods=days * 24, freq="h")
    demand = np.arange(len(ts))
    return pd.DataFrame(
        {
            "instant": demand + 1,
            "dteday": ts.strftime("%Y-%m-%d"),
            "season": 1,
            "yr": 0,
            "mnth": ts.month,
            "hr": ts.hour,
            "holiday": 0,
            "weekday": (ts.day_of_week + 1) % 7,
            "workingday": (ts.dayofweek < 5).astype(int),
            "weathersit": 1,
            "temp": 0.5,
            "atemp": 0.5,
            "hum": 0.5,
            "windspeed": 0.1,
            "casual": 0,
            "registered": demand,
            "cnt": demand,
        }
    )


def test_unit_cutoff_features_known_values() -> None:
    """Features for day 8 at 10:00 (t = 202) match the hand-computed values."""
    out = clean_dataframe(_make_hourly_df(10), target_column="cnt", cutoff_hour=17)
    row = out.loc[out["ts"] == pd.Timestamp("2011-01-09 10:00")].iloc[0]
    assert row["cnt"] == 202
    assert row["lag_48"] == 154
    assert row["lag_168"] == 34
    assert row["mean_168_to_cutoff"] == pytest.approx(100.5)


def test_unit_training_mode_drops_warm_up_rows() -> None:
    """Training mode keeps only rows with full history: 48 of 240, from day 8 on."""
    out = clean_dataframe(_make_hourly_df(10), target_column="cnt", cutoff_hour=17)
    assert len(out) == 48
    assert out["ts"].min() == pd.Timestamp("2011-01-09 00:00")
    assert out[["lag_48", "lag_168", "mean_168_to_cutoff"]].notna().all().all()


def test_unit_missing_hour_inserted_and_input_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    """A missing hour is inserted with zero demand and its date's flags, and the input frame is untouched."""
    df = _make_hourly_df(10)
    df.loc[df["dteday"] == "2011-01-03", ["holiday", "workingday"]] = [1, 0]
    df_raw = df.drop(index=58)
    before = df_raw.copy()
    with caplog.at_level(logging.WARNING, logger="src.clean_data"):
        out = clean_dataframe(df_raw, target_column=None)
    row = out.loc[out["ts"] == pd.Timestamp("2011-01-03 10:00")].iloc[0]
    assert len(out) == 240
    assert row[["cnt", "casual", "registered"]].tolist() == [0, 0, 0]
    assert row["is_imputed"] == 1
    assert row[["hr", "weekday", "holiday", "workingday"]].tolist() == [10, 1, 1, 0]
    pd.testing.assert_frame_equal(df_raw, before)
    assert "Inserted 1 missing hours" in caplog.text


@pytest.mark.parametrize("date, expected_workingday", [("2011-01-08", 0), ("2011-01-05", 1)])
def test_unit_missing_whole_date_uses_weekday_rule(date: str, expected_workingday: int) -> None:
    """A fully missing date gets holiday 0 and workingday 1 only Monday to Friday."""
    df = _make_hourly_df(10)
    out = clean_dataframe(df[df["dteday"] != date], target_column=None)
    inserted = out[out["ts"].dt.strftime("%Y-%m-%d") == date]
    assert len(inserted) == 24
    assert (inserted["is_imputed"] == 1).all()
    assert (inserted["holiday"] == 0).all()
    assert (inserted["workingday"] == expected_workingday).all()


def test_unit_inference_mode_keeps_every_row() -> None:
    """Inference mode keeps every row, including warm-up rows and tomorrow's rows with empty demand."""
    df = _make_hourly_df(11)
    df.loc[df.index >= 240, ["casual", "registered", "cnt"]] = np.nan
    out = clean_dataframe(df, target_column=None)
    assert len(out) == 264
    assert out["cnt"].iloc[240:].isna().all()
    assert (out["is_imputed"] == 0).all()


def test_unit_no_leakage_after_cutoff() -> None:
    """Scrambling demand from 17:00 on D - 1 onwards leaves every feature of date D unchanged."""
    df = _make_hourly_df(10)
    features = ["lag_48", "lag_168", "mean_168_to_cutoff"]
    target_day = pd.Timestamp("2011-01-10")
    clean = clean_dataframe(df, target_column=None)
    scrambled = df.copy()
    raw_ts = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    after_cutoff = raw_ts >= target_day - pd.Timedelta(hours=7)
    scrambled.loc[after_cutoff, "cnt"] = np.random.default_rng(42).integers(0, 1000, after_cutoff.sum())
    leaked = clean_dataframe(scrambled, target_column=None)
    on_d = clean["ts"].dt.normalize() == target_day
    pd.testing.assert_frame_equal(clean.loc[on_d, features], leaked.loc[on_d, features])


def _broken_frame(case: str) -> pd.DataFrame:
    """Build a 2-day frame broken in the way named by case.

    Inputs: case, one of empty, missing_column, bad_date, duplicate_ts or valid.
    Outputs: DataFrame that should trigger the matching error rule.
    Why: one builder keeps the parametrized error test to a single short body.
    """
    df = _make_hourly_df(2)
    if case == "empty":
        return df.iloc[0:0]
    if case == "missing_column":
        return df.drop(columns=["hr"])
    if case == "bad_date":
        return df.assign(dteday=df["dteday"].where(df.index != 5, "not-a-date"))
    if case == "duplicate_ts":
        return pd.concat([df, df.iloc[[3]]])
    return df


@pytest.mark.parametrize(
    "case, kwargs, match",
    [
        ("empty", {}, "empty"),
        ("missing_column", {}, r"\['hr'\].*demand_column"),
        ("valid", {"target_column": "rentals"}, r"rentals.*target_column"),
        ("bad_date", {}, "not-a-date"),
        ("duplicate_ts", {}, "2011-01-01 03:00"),
        ("valid", {"cutoff_hour": 0}, r"0.*cutoff_hour"),
        ("valid", {"cutoff_hour": 24}, r"24.*cutoff_hour"),
    ],
)
def test_unit_input_errors(case: str, kwargs: dict, match: str) -> None:
    """Each broken input raises ValueError naming the value and the SETTINGS key to check."""
    with pytest.raises(ValueError, match=match):
        clean_dataframe(_broken_frame(case), **kwargs)
