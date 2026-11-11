"""Shared read and write helpers for CSV files and the trained model.

Serves README section 1, "Value, cost and risks" (risk: data quality): every module
reads and writes through these four tested helpers, so file handling lives in one place.
Generic only: no column names, no date parsing, no business rules.
Paths always arrive as parameters from SETTINGS in src/main.py.
"""
import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame without changing any values.

    Inputs: filepath, path to the CSV file (str or Path).
    Outputs: the file contents as a DataFrame; text columns such as timestamps stay text.
    Raises: FileNotFoundError if no file exists at filepath.
    Why: one tested reader for every module; parsing dates belongs to clean_data.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}. Check the path set in SETTINGS['paths'].")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Write a DataFrame to CSV, creating missing parent folders.

    Inputs: df, the DataFrame to write; filepath, target CSV path (str or Path).
    Outputs: none; the file is written to filepath without the index.
    Raises: TypeError if df is not a pandas DataFrame.
    Why: one tested writer for every module; index=False avoids an 'Unnamed: 0'
    column that would break the required-column check in validate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"save_csv expects a pandas DataFrame, got {type(df).__name__}. "
            "Check the value returned by the previous pipeline step."
        )
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %d rows to %s", len(df), path)


def save_model(model: object, filepath: Path) -> None:
    """Save a fitted model to disk with joblib, creating missing parent folders.

    Inputs: model, any fitted model or pipeline; filepath, target file path (str or Path).
    Outputs: none; the model is written to filepath.
    Why: train saves and infer reloads the exact same file, so tomorrow's forecast
    uses the model that passed the quality gate.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(filepath: Path) -> object:
    """Load a model saved with save_model.

    Inputs: filepath, path to the saved model file (str or Path).
    Outputs: the model object, ready to call predict on.
    Raises: FileNotFoundError if no file exists at filepath.
    Why: infer reloads the exact model that train saved and the gate approved.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run 'python -m src.main' to train and save it, "
            "or check SETTINGS['paths']."
        )
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model
