# imports
import os
import pickle

import numpy as np
import pandas as pd
from joblib import dump
from loguru import logger
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from utils.config import MODELS_PATH
from utils.config import PROCESSED_DATA_FOLDER


def train_model(**kwargs):
    params = kwargs['params']
    np.random.seed(params['seed'])
    logger.info(f"Training model")

    # Read Data
    logger.info(f"Params passed to train: {kwargs}")
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_FOLDER, 'y_train.csv'))

    # Define model and grid search
    pipe = Pipeline([('scale', StandardScaler()),
                     ('selector', SelectKBest(mutual_info_regression)),
                     ('poly', PolynomialFeatures()),
                     ('model', Ridge())])
    logger.info(f"Model to train: {pipe}")

    k = params['k']
    alpha = params['alpha']
    poly = params['poly']

    grid = GridSearchCV(estimator=pipe,
                        param_grid=dict(selector__k=k,
                                        poly__degree=poly,
                                        model__alpha=alpha),
                        cv=3,
                        scoring='r2')

    logger.info(f"Grid Search with {grid.param_grid}")

    # Train model
    grid.fit(X_train, y_train)
    logger.info(f"Grid Search Finished. Best Params: {grid.best_params_}")

    # Save Model
    dump(grid, os.path.join(MODELS_PATH, "ridge-regression-model.joblib"))
    pickle.dump(grid, open(os.path.join(MODELS_PATH, "ridge-regression-model.pkl"), 'wb'))
