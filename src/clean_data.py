"""Hourly calendar and cutoff-safe past-demand features for the bike demand forecast.

Serves README "1. Business case": Solution (forecast tomorrow at 18:00 with data up to 17:00)
and Data (missing hours mean no rentals were recorded). Fixed rules only, nothing learned
from the data, so it is safe before the split. Every feature for a target date D uses only
demand up to hour cutoff_hour - 1 on D - 1; tests/test_clean_data.py proves this.
Called by main.py in training and inference mode; next step: validate.py.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _check_inputs(df: pd.DataFrame, target_column: str | None, cutoff_hour: int, demand_column: str) -> None:
    """Stop early when the raw frame or the cleaning settings cannot be used.

    Inputs: df, the raw hourly frame; target_column, the target or None in inference mode;
        cutoff_hour, the first hour on D - 1 that features may not use; demand_column, the column lags read.
    Outputs: none.
    Raises: ValueError for an empty frame, missing columns, an unknown target or a cutoff outside 1 to 23.
    Why: each problem would otherwise surface later as a confusing pandas error or as silent leakage.
    """
    if df.empty:
        raise ValueError("The raw data is empty. Check SETTINGS['paths']['raw_data'] and the raw data export.")
    missing = [column for column in ["dteday", "hr", demand_column] if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Check SETTINGS['cleaning']['demand_column'] and the raw data export."
        )
    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' is not in the data. Check SETTINGS['problem']['target_column']."
        )
    if not 1 <= cutoff_hour <= 23:
        raise ValueError(
            f"cutoff_hour must be between 1 and 23, got {cutoff_hour}. Check SETTINGS['cleaning']['cutoff_hour']."
        )


def _build_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Add ts = dteday + hr hours, sorted, and stop on unparseable dates or duplicate hours.

    Inputs: df, the working copy of the raw frame with dteday and hr.
    Outputs: the same rows with a ts column, sorted by ts, index reset.
    Raises: ValueError if dteday cannot be parsed or two rows share the same ts.
    Why: every later step works by clock time, which needs one unique, ordered timestamp per row.
    """
    try:
        dates = pd.to_datetime(df["dteday"])
    except ValueError as err:
        bad = df.loc[pd.to_datetime(df["dteday"], errors="coerce").isna(), "dteday"].iloc[0]
        raise ValueError(f"Column 'dteday' has unparseable dates, e.g. '{bad}'. Check the raw data export.") from err
    df["ts"] = dates + pd.to_timedelta(df["hr"], unit="h")
    duplicated = df["ts"].duplicated()
    if duplicated.any():
        example = df.loc[duplicated, "ts"].iloc[0]
        raise ValueError(f"Duplicate hour in the raw data, e.g. {example}. Check the raw data export.")
    return df.sort_values("ts").reset_index(drop=True)


def _complete_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Insert every missing hour between the first and last ts and flag it with is_imputed.

    Inputs: df, the sorted frame with a unique ts column.
    Outputs: one row per hour from the first to the last ts; is_imputed is 1 on inserted rows, 0 elsewhere,
        and the other columns of inserted rows are still empty (filled by _fill_calendar_columns).
    Why: shift and rolling count rows, so rows must equal hours before any past-demand feature is built.
    """
    df["is_imputed"] = 0
    full_range = pd.date_range(df["ts"].iloc[0], df["ts"].iloc[-1], freq="h")
    completed = df.set_index("ts").reindex(full_range).rename_axis("ts").reset_index()
    completed["is_imputed"] = completed["is_imputed"].fillna(1).astype(int)
    inserted = int(completed["is_imputed"].sum())
    if inserted > 0:
        logger.warning("Inserted %d missing hours (is_imputed=1)", inserted)
    return completed


def _fill_calendar_columns(df: pd.DataFrame, demand_column: str) -> pd.DataFrame:
    """Give inserted hours zero demand and rebuild the calendar columns from ts.

    Inputs: df, the completed calendar; demand_column, the demand column set to 0 on inserted rows.
    Outputs: the same frame with demand, casual and registered 0 on inserted rows, hr, mnth, weekday and
        dteday from ts, and season and yr carried forward from the previous existing hour.
    Why: README Data assumes a missing hour had no rentals; only inserted rows get 0, so tomorrow's
        empty demand in inference mode stays missing.
    """
    imputed = df["is_imputed"] == 1
    df.loc[imputed, [demand_column, "casual", "registered"]] = 0
    df["dteday"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["hr"] = df["ts"].dt.hour
    df["mnth"] = df["ts"].dt.month
    df["weekday"] = (df["ts"].dt.day_of_week + 1) % 7
    df[["season", "yr"]] = df[["season", "yr"]].ffill().astype(int)
    return df


def _fill_day_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Fill holiday and workingday on inserted hours from the same date, else from the weekday rule.

    Inputs: df, the completed calendar with ts, holiday and workingday.
    Outputs: the same frame with holiday and workingday as integers on every row.
    Why: a date keeps one holiday status; a fully missing date gets holiday 0 and workingday 1
        Monday to Friday (notebook section 2).
    """
    date = df["ts"].dt.normalize()
    for column in ["holiday", "workingday"]:
        df[column] = df[column].fillna(df.groupby(date)[column].transform("first"))
    df["holiday"] = df["holiday"].fillna(0).astype(int)
    df["workingday"] = df["workingday"].fillna((df["ts"].dt.day_of_week < 5).astype(int)).astype(int)
    return df


def _add_past_demand_features(df: pd.DataFrame, demand_column: str, cutoff_hour: int) -> pd.DataFrame:
    """Add the five past-demand features, each using only hours before cutoff_hour on D - 1.

    Inputs: df, the completed calendar sorted by ts; demand_column, the demand to read;
        cutoff_hour, the first hour on D - 1 that features may not use.
    Outputs: the same frame with lag_48, lag_168, mean_168_to_cutoff (missing if fewer than 168 hours),
        lag_24_before_cutoff (same hour yesterday, missing from cutoff_hour on) and
        last_hour_before_cutoff (demand at hour cutoff_hour - 1 on D - 1).
    Why: the forecast is made at the cutoff on D - 1, so every feature may use only hours before it;
        the two recent-level features lower the validation ratio from 1.05 to 0.92
        (notebooks/max_iter_and_gate.ipynb).
    """
    demand = df[demand_column]
    df["lag_48"] = demand.shift(48)
    df["lag_168"] = demand.shift(168)
    df["lag_24_before_cutoff"] = demand.shift(24).where(df["hr"] < cutoff_hour)
    window_mean = pd.Series(demand.rolling(168, min_periods=168).mean().to_numpy(), index=df["ts"])
    window_end = df["ts"].dt.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=cutoff_hour - 1)
    df["mean_168_to_cutoff"] = window_end.map(window_mean)
    df["last_hour_before_cutoff"] = window_end.map(pd.Series(demand.to_numpy(), index=df["ts"]))
    return df


def _drop_incomplete_rows(df: pd.DataFrame, target_column: str | None) -> pd.DataFrame:
    """In training mode, drop rows without a target or without full past-demand history.

    Inputs: df, the frame with past-demand features; target_column, the target or None in inference mode.
    Outputs: the kept rows with a fresh index; in inference mode, df unchanged.
    Why: training needs complete rows (brief: 7 days of history); inference must keep tomorrow's rows.
    """
    if target_column is None:
        return df
    required = [target_column, "lag_48", "lag_168", "mean_168_to_cutoff"]
    kept = df.dropna(subset=required).reset_index(drop=True)
    logger.info("Dropped %d rows without full history", len(df) - len(kept))
    return kept


def clean_dataframe(
    df_raw: pd.DataFrame, target_column: str | None = None, cutoff_hour: int = 17, demand_column: str = "cnt"
) -> pd.DataFrame:
    """Turn raw hourly rows into a complete hourly calendar with cutoff-safe past-demand features.

    Inputs: df_raw, the raw hourly frame (hour.csv layout); target_column, the target in training mode,
        None in inference mode; cutoff_hour, from SETTINGS['cleaning']['cutoff_hour'];
        demand_column, from SETTINGS['cleaning']['demand_column'].
    Outputs: a new frame with all original columns plus ts, is_imputed and the five past-demand features
        (see _add_past_demand_features).
    Raises: ValueError for any broken input (see _check_inputs and _build_timestamp).
    Why: fixed past-only rules, so the 17:00 cutoff from README Solution holds before the split.
    """
    _check_inputs(df_raw, target_column, cutoff_hour, demand_column)
    df = _build_timestamp(df_raw.copy())
    df = _complete_calendar(df)
    df = _fill_calendar_columns(df, demand_column)
    df = _fill_day_flags(df)
    # Inference history ends at D - 1, hour cutoff_hour - 1, so the later hours of D - 1 are inserted with
    # demand 0. No feature of date D reads them (test_unit_no_leakage_after_cutoff), and main predicts only
    # rows with missing demand.
    df = _add_past_demand_features(df, demand_column, cutoff_hour)
    df = _drop_incomplete_rows(df, target_column)
    logger.info("Cleaned %d raw rows into %d rows", len(df_raw), len(df))
    return df
