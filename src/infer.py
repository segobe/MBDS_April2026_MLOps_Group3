"""Step 7 (infer): tomorrow's hourly forecast from the saved Pipeline.

Serves README section 1, "Solution": the Operations Manager plans crew hours from a 24-hour
forecast produced the evening before. Only predict, never fit: the saved Pipeline runs the same
steps as training, so the forecast is the one that passed the gate. Predictions are clipped at 0
(same rule as evaluate) and keep the input index; main selects tomorrow's rows and adds ts back.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _check_inference_inputs(model: Pipeline, X_infer: pd.DataFrame, include_proba: bool) -> None:
    """Stop when the model cannot predict, there are no rows, or probabilities are requested.

    Inputs: model, the loaded model; X_infer, tomorrow's feature rows; include_proba, the caller's flag.
    Outputs: none.
    Raises: TypeError if model has no predict method; ValueError if X_infer is empty or include_proba is True.
    Why: each problem gets its own clear message instead of a confusing scikit-learn error at 18:00.
    """
    if not hasattr(model, "predict"):
        raise TypeError(
            f"model has no predict method, got {type(model).__name__}. "
            "Check that SETTINGS['paths']['model'] points to the file saved by train."
        )
    if X_infer.empty:
        raise ValueError(
            "No rows to forecast: the inference data is empty. "
            "Check the inference file at SETTINGS['paths']['inference_data']."
        )
    if include_proba:
        raise ValueError(
            "include_proba=True is not supported: a regression model has no probabilities. "
            "Call run_inference with include_proba=False."
        )


def run_inference(model: Pipeline, X_infer: pd.DataFrame, include_proba: bool = False) -> pd.DataFrame:
    """Predict demand for each row of X_infer, clipped at 0, keeping the input index.

    Inputs: model, the fitted Pipeline from utils.load_model; X_infer, tomorrow's feature rows;
        include_proba, must stay False for this regression model.
    Outputs: DataFrame with one float column 'prediction' and the same index as X_infer.
    Raises: TypeError or ValueError from _check_inference_inputs.
    Why: the planner needs one non-negative forecast per hour, joinable back to its ts by index.
    """
    _check_inference_inputs(model, X_infer, include_proba)
    predictions = np.clip(model.predict(X_infer), 0, None)  # no negative rentals
    result = pd.DataFrame({"prediction": predictions.astype(float)}, index=X_infer.index)
    logger.info(
        "Predicted %d rows, min %.1f, max %.1f", len(result), result["prediction"].min(), result["prediction"].max()
    )
    return result
