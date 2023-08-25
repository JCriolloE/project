import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(parent_dir, ".."))
sys.path.append(parent_dir)

from joblib import load
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)

class Prediction:

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(X_val):
        """
        Loading the model and predicting with validation set
        """
        regression_model = load("models/regression.joblib")

        y_pred_log_reg_val = regression_model.predict(X_val)

        logger.info(f"Prediction results: {y_pred_log_reg_val}")
        return (y_pred_log_reg_val)
    
class V2Prediction:

    def __init__(self, model_path):
        self.model_path = model_path

    def predict(X_val):
        """
        Loading the model and predicting with validation set
        """
        regression_model = load("models/regression2.joblib")

        y_pred_log_reg_val = regression_model.predict(X_val)

        logger.info(f"Prediction results: {y_pred_log_reg_val}")
        return (y_pred_log_reg_val)
