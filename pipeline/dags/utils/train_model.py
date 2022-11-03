# imports
import os
import pickle

import pandas as pd
from joblib import dump
from loguru import logger
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from utils.config import PROCESSED_DATA_FOLDER


def train_model():
    logger.info(f"Training model")
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv'))

    pipe = Pipeline([('scale', StandardScaler()),
                     ('selector', SelectKBest(mutual_info_regression)),
                     ('poly', PolynomialFeatures()),
                     ('model', Ridge())])
    logger.info(f"Model to train: {pipe}")

    k = [3, 4, 5, 6, 7, 10]
    alpha = [1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    poly = [1, 2, 3, 5, 7]
    grid = GridSearchCV(estimator=pipe,
                        param_grid=dict(selector__k=k,
                                        poly__degree=poly,
                                        model__alpha=alpha),
                        cv=3,
                        scoring='r2')

    logger.info(f"Grid Search with {grid.param_grid}")
    grid.fit(X_train, y_train)
    logger.info(f"Grid Search Finished. Best Params: {grid.best_params_}")

    dump(grid, 'models/ridge-regression-model.joblib')
    pickle.dump(grid, open('models/ridge-regression-model.pkl', 'wb'))
