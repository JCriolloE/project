import os
import sys

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utilities import custom_logging

logger = custom_logging.CustomLogging(__name__)


def training(X_train, y_train):
    # Let's implement simple classifiers
    {
        "LogisiticRegression": LogisticRegression()
    }

    # Logistic Regression
    log_reg_params = {
        "penalty": [
            'l1', 'l2'], 'C': [
            0.001, 0.01, 0.1, 1, 10, 100, 1000]}

    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_log_reg.fit(X_train, y_train)
    # We automatically get the logistic regression with the best parameters.
    log_reg = grid_log_reg.best_estimator_

    """
        Saving the best model
    """
    dump(log_reg,   "models/regression.joblib")

    logger.info("Model trained and saved")
    return log_reg
