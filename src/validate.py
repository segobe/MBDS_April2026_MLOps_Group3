"""Step 3 (validate): the data quality gate after clean_data, in training and inference mode.

Serves README section 1, "Value, cost and risks" (risk: data quality): automated schema
validation, missing-value checks and duplicate detection. Validates, never fixes: returns True
or raises ValueError naming the rule, the column and an example value.
Every column name and limit arrives as a parameter from SETTINGS['validation'] in src/main.py.
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _check_not_empty(df: pd.DataFrame) -> None:
    """Stop when the frame has no rows.

    Inputs: df, the cleaned frame.
    Outputs: none.
    Raises: ValueError if df has no rows.
    Why: every later rule would pass trivially on an empty frame.
    """
    if df.empty:
        raise ValueError("The data to validate is empty. Check the clean_data output and SETTINGS['paths'].")


def _check_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    """Stop when any required column is missing, listing all of them.

    Inputs: df, the cleaned frame; required_columns, the column names that must be present.
    Outputs: none.
    Raises: ValueError listing every missing column.
    Why: one message with the full list saves a fix-and-rerun loop per column.
    """
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing}. "
            "Check the clean_data output and SETTINGS['validation']['required_columns']."
        )


def _check_unique_id(df: pd.DataFrame, id_column: str | None) -> None:
    """Stop when the id column is absent or has duplicate values.

    Inputs: df, the cleaned frame; id_column, the column that identifies each row, or None to skip.
    Outputs: none.
    Raises: ValueError if id_column is not in df or has a duplicate value.
    Why: a duplicate hour would count demand twice in training and in the forecast.
    """
    if id_column is None:
        return
    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' is not in the data. Check SETTINGS['validation']['id_column'].")
    duplicates = df.loc[df[id_column].duplicated(), id_column]
    if not duplicates.empty:
        raise ValueError(
            f"Column '{id_column}' has duplicate values, e.g. {duplicates.iloc[0]}. Check the clean_data output."
        )


def _check_target(df: pd.DataFrame, target_column: str | None) -> None:
    """In training mode, stop when the target is absent or has missing values.

    Inputs: df, the cleaned frame; target_column, the target name, or None in inference mode.
    Outputs: none.
    Raises: ValueError if target_column is not in df or has a missing value.
    Why: the model cannot learn from rows without a known demand.
    """
    if target_column is None:
        return
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' is not in the data. Check SETTINGS['problem']['target_column']."
        )
    missing_rows = df.index[df[target_column].isna()]
    if len(missing_rows) > 0:
        raise ValueError(
            f"Target column '{target_column}' has {len(missing_rows)} missing values, e.g. row {missing_rows[0]}. "
            "Check the clean_data output."
        )


def _rule_columns(
    value_ranges: dict | None, allowed_values: dict | None, integer_columns: list[str] | None, sum_rules: dict | None
) -> list[tuple[str, str]]:
    """List every (rule name, column) pair the rules refer to.

    Inputs: value_ranges, allowed_values, integer_columns, sum_rules, the rules (None = no rule).
    Outputs: list of (rule name, column name) pairs; sum rules contribute the total and every part.
    Why: lets one check confirm that every column a rule names exists.
    """
    pairs = [("value_ranges", column) for column in value_ranges or {}]
    pairs += [("allowed_values", column) for column in allowed_values or {}]
    pairs += [("integer_columns", column) for column in integer_columns or []]
    for total, parts in (sum_rules or {}).items():
        pairs += [("sum_rules", column) for column in [total, *parts]]
    return pairs


def _check_rule_columns_exist(df: pd.DataFrame, rule_columns: list[tuple[str, str]]) -> None:
    """Stop when a rule names a column that is not in the data.

    Inputs: df, the cleaned frame; rule_columns, (rule name, column) pairs from _rule_columns.
    Outputs: none.
    Raises: ValueError naming the rule and the unknown column.
    Why: a typo in SETTINGS would otherwise surface as a confusing KeyError.
    """
    for rule, column in rule_columns:
        if column not in df.columns:
            raise ValueError(
                f"Rule '{rule}' names column '{column}', which is not in the data. "
                f"Check SETTINGS['validation']['{rule}']."
            )


def _outside_range(values: pd.Series, low: float | None, high: float | None) -> pd.Series:
    """Return the values below low or above high, treating None as no limit.

    Inputs: values, a series without missing values; low and high, the limits or None.
    Outputs: the offending values (empty if all are in range).
    Why: each side of a range can be open, so each is checked on its own.
    """
    outside = pd.Series(False, index=values.index)
    if low is not None:
        outside |= values < low
    if high is not None:
        outside |= values > high
    return values[outside]


def _check_ranges(df: pd.DataFrame, value_ranges: dict | None) -> None:
    """Stop when a column has values outside its [low, high] range, ignoring missing values.

    Inputs: df, the cleaned frame; value_ranges, {column: [low, high]} with None = no limit, or None.
    Outputs: none.
    Raises: ValueError naming the column, the range and an example value.
    Why: an hour of 24 or a negative demand means the data is broken; tomorrow's empty cnt is not.
    """
    for column, (low, high) in (value_ranges or {}).items():
        bad = _outside_range(df[column].dropna(), low, high)
        if not bad.empty:
            raise ValueError(
                f"Column '{column}' has values outside [{low}, {high}], e.g. {bad.iloc[0]}. "
                "Check the raw data export or SETTINGS['validation']['value_ranges']."
            )


def _check_allowed_values(df: pd.DataFrame, allowed_values: dict | None) -> None:
    """Stop when a column has a value outside its allowed set, ignoring missing values.

    Inputs: df, the cleaned frame; allowed_values, {column: [allowed values]}, or None.
    Outputs: none.
    Raises: ValueError naming the column, the allowed set and an example value.
    Why: flags such as holiday must be exactly 0 or 1 for the model to read them correctly.
    """
    for column, allowed in (allowed_values or {}).items():
        values = df[column].dropna()
        bad = values[~values.isin(allowed)]
        if not bad.empty:
            raise ValueError(
                f"Column '{column}' has values not in {allowed}, e.g. {bad.iloc[0]}. "
                "Check the raw data export or SETTINGS['validation']['allowed_values']."
            )


def _check_integers(df: pd.DataFrame, integer_columns: list[str] | None) -> None:
    """Stop when a column holds non-whole numbers, ignoring missing values.

    Inputs: df, the cleaned frame; integer_columns, columns that must hold whole numbers, or None.
    Outputs: none.
    Raises: ValueError naming the column and an example value.
    Why: rentals are counted, so 2.5 bikes means a broken export.
    """
    for column in integer_columns or []:
        values = df[column].dropna()
        bad = values[values % 1 != 0]
        if not bad.empty:
            raise ValueError(
                f"Column '{column}' must hold whole numbers (integer), e.g. {bad.iloc[0]}. "
                "Check the raw data export or SETTINGS['validation']['integer_columns']."
            )


def _check_sums(df: pd.DataFrame, sum_rules: dict | None) -> None:
    """Stop when a total column differs from the sum of its parts.

    Inputs: df, the cleaned frame; sum_rules, {total: [part, ...]}, or None to skip.
    Outputs: none.
    Raises: ValueError naming the total, the parts and an example row.
    Why: cnt must equal casual + registered, otherwise the target itself is wrong.
    """
    for total, parts in (sum_rules or {}).items():
        parts_sum = df[parts].sum(axis=1)
        bad_rows = df.index[df[total] != parts_sum]
        if len(bad_rows) > 0:
            row = bad_rows[0]
            raise ValueError(
                f"Column '{total}' is not the sum of {parts}, e.g. row {row}: {df.loc[row, total]} "
                f"vs {parts_sum[row]}. Check the raw data export or SETTINGS['validation']['sum_rules']."
            )


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    target_column: str | None = None,
    id_column: str | None = None,
    value_ranges: dict | None = None,
    allowed_values: dict | None = None,
    integer_columns: list[str] | None = None,
    sum_rules: dict | None = None,
) -> bool:
    """Check the cleaned data against every rule and stop at the first broken one.

    Inputs: df, the cleaned frame; required_columns, columns that must exist; target_column, the target
        in training mode or None in inference mode; id_column, value_ranges, allowed_values,
        integer_columns, sum_rules, the rules from SETTINGS['validation'] (None = skip that rule).
    Outputs: True when every rule passes; df is never modified.
    Raises: ValueError naming the broken rule, the column and an example value.
    Why: bad data must stop the batch before training or forecasting (README risk: data quality).
    """
    active_sums = sum_rules if target_column is not None else None
    _check_not_empty(df)
    _check_required_columns(df, required_columns)
    _check_unique_id(df, id_column)
    _check_target(df, target_column)
    _check_rule_columns_exist(df, _rule_columns(value_ranges, allowed_values, integer_columns, active_sums))
    _check_ranges(df, value_ranges)
    _check_allowed_values(df, allowed_values)
    _check_integers(df, integer_columns)
    _check_sums(df, active_sums)
    logger.info("Validation passed: %d rows", len(df))
    return True
