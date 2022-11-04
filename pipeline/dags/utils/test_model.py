# imports
import os

import mlflow
import pandas as pd
from joblib import load
from loguru import logger
from mlflow import log_metric, log_param
from sklearn.metrics import mean_squared_error, r2_score
from utils.config import MODELS_PATH
from utils.config import PROCESSED_DATA_FOLDER


def test_model(**kwargs):
    params = kwargs['params']
    logger.info("Testing Model")

    # Paths
    model_path = os.path.join(MODELS_PATH, "ridge-regression-model.joblib")
    x_test_path = os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv')
    y_test_path = os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv')
    logger.info(f"Loading Model {model_path}")
    logger.info(f"Test data: X={x_test_path} , y={y_test_path}")

    # Load data and model
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    clf = load(model_path)

    # Save params
    mlflow.set_experiment(params['exp_name'])
    for param, val in params.items():
        if params == 'exp_name':
            continue
        log_param(param, val)

    for param, val in clf.best_params_.items():
        log_param(param, val)

    # Eval model and log metrics
    y_predicted = clf.predict(X_test)
    rmse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)
    log_metric('RMSE', rmse)
    log_metric("R2", r2)

    # Log values
    logger.info(f'RMSE: {rmse}')
    logger.info(f'R2: {r2}')
