"""Step 1 (load): read the raw hourly rental file and stop early on bad input.

Serves README section 1, "Value, cost and risks" (risk: data quality): the nightly
batch fails fast with a clear message instead of producing a silent wrong forecast.
Reads only, through utils.load_csv: no column checks (validate), no date parsing (clean_data).
The path arrives as a parameter from SETTINGS['paths']['raw_data'] in src/main.py.
"""
import logging
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)


def _check_path(path: Path) -> None:
    """Stop when the raw-data path is missing or points to a folder.

    Inputs: path, the raw data path already converted to Path.
    Outputs: none.
    Raises: FileNotFoundError if nothing exists at path; IsADirectoryError if path is a folder.
    Why: each problem gets its own clear message before pandas is ever called.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}. Check SETTINGS['paths']['raw_data'].")
    if path.is_dir():
        raise IsADirectoryError(
            f"Raw data path is a folder, not a file: {path}. Check SETTINGS['paths']['raw_data']."
        )


def _read_non_empty(path: Path) -> pd.DataFrame:
    """Read the raw CSV through utils.load_csv and stop when it has no data rows.

    Inputs: path, an existing file path (already checked by _check_path).
    Outputs: the raw DataFrame with at least one row.
    Raises: ValueError if the file is 0 bytes or has a header but no rows.
    Why: an empty export must stop the batch, not train or forecast on nothing.
    """
    try:
        df = utils.load_csv(path)
    except pd.errors.EmptyDataError as err:
        raise ValueError(f"Raw data file is empty (0 bytes): {path}. Check SETTINGS['paths']['raw_data'].") from err
    if len(df) == 0:
        raise ValueError(
            f"Raw data file has a header but no rows: {path}. Check SETTINGS['paths']['raw_data']."
        )
    return df


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Load the raw hourly rental file exactly as it is on disk.

    Inputs: raw_data_path, path to the raw CSV (str or Path), from SETTINGS['paths']['raw_data'].
    Outputs: the raw DataFrame, unchanged (no column checks, no date parsing).
    Raises: FileNotFoundError, IsADirectoryError or ValueError for a missing, folder or empty input.
    Why: the first pipeline step fails fast, so bad input never reaches training or the forecast.
    """
    path = Path(raw_data_path)
    _check_path(path)
    df = _read_non_empty(path)
    logger.info("Loaded %d rows and %d columns from %s", len(df), df.shape[1], path)
    return df
