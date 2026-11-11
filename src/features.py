"""Step 4 (features): the unfitted recipe that decides which columns reach the model.

Serves README section 1, "Data", and the brief's excluded columns: only calendar variables and
cutoff-safe past demand pass; everything else (casual, registered, weather, is_imputed, ts,
dteday, instant, yr) is dropped, so it can never leak into training.
Column lists arrive as parameters from SETTINGS['features'] in src/main.py; train fits the recipe.
"""
import logging

from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


def _check_column_list(columns: list[str], group: str) -> None:
    """Stop when a feature group is empty or lists a column twice.

    Inputs: columns, the column names of one group; group, its SETTINGS key, used in the message.
    Outputs: none.
    Raises: ValueError if columns is empty or contains a duplicate.
    Why: an empty group silently removes a whole feature family; a duplicate feeds one column twice.
    """
    if not columns:
        raise ValueError(f"Feature group '{group}' is empty. Check SETTINGS['features']['{group}'].")
    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        raise ValueError(
            f"Feature group '{group}' lists a column twice: '{duplicates[0]}'. Check SETTINGS['features']['{group}']."
        )


def _check_no_overlap(calendar_columns: list[str], past_demand_columns: list[str]) -> None:
    """Stop when a column appears in both feature groups.

    Inputs: calendar_columns and past_demand_columns, the two groups.
    Outputs: none.
    Raises: ValueError naming the first overlapping column.
    Why: a column in both groups would reach the model twice.
    """
    overlap = sorted(set(calendar_columns) & set(past_demand_columns))
    if overlap:
        raise ValueError(
            f"Column '{overlap[0]}' is in both calendar_columns and past_demand_columns. "
            "Check SETTINGS['features']."
        )


def get_feature_preprocessor(calendar_columns: list[str], past_demand_columns: list[str]) -> ColumnTransformer:
    """Build the unfitted feature recipe: two passthrough groups, every other column dropped.

    Inputs: calendar_columns, e.g. hr and weekday; past_demand_columns, e.g. lag_48 and lag_168.
    Outputs: an unfitted ColumnTransformer; train fits it on the training split only.
    Raises: ValueError if a group is empty, lists a column twice, or shares a column with the other.
    Why: remainder="drop" enforces the brief's excluded columns in code.
    """
    _check_column_list(calendar_columns, "calendar_columns")
    _check_column_list(past_demand_columns, "past_demand_columns")
    _check_no_overlap(calendar_columns, past_demand_columns)
    # Passthrough: HistGradientBoostingRegressor handles raw integers and missing values itself,
    # and the notebook used no preprocessing. remainder="drop" keeps casual and registered out.
    recipe = ColumnTransformer(
        [("calendar", "passthrough", calendar_columns), ("past_demand", "passthrough", past_demand_columns)],
        remainder="drop",
    )
    logger.info(
        "Feature recipe: %d calendar and %d past-demand columns", len(calendar_columns), len(past_demand_columns)
    )
    return recipe


# Recommendations for later (not built now):
# - Treat hr and weekday as categorical features (HistGradientBoostingRegressor categorical_features).
# - Holiday-aware lags: compare a holiday with the previous holiday, not with the same weekday last week.
