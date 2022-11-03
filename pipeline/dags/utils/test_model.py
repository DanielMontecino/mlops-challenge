# imports
import os

import pandas as pd
from joblib import load
from loguru import logger
from sklearn.metrics import mean_squared_error, r2_score
from utils.config import PROCESSED_DATA_FOLDER


def test_model():
    logger.info("Testing Model")
    model_path = 'models/ridge-regression-model.joblib'
    x_test_path = os.path.join(PROCESSED_DATA_FOLDER, 'X_test.csv')
    y_test_path = os.path.join(PROCESSED_DATA_FOLDER, 'y_test.csv')
    logger.info(f"Loading Model {model_path}")
    logger.info(f"Test data: X={x_test_path} , y={y_test_path}")
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)
    clf = load(model_path)

    y_predicted = clf.predict(X_test)

    # evaluar modelo
    rmse = mean_squared_error(y_test, y_predicted)
    r2 = r2_score(y_test, y_predicted)

    # printing values
    logger.info(f'RMSE: {rmse}')
    logger.info(f'R2: {r2}')
