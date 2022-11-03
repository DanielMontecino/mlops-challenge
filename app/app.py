import pickle

from fastapi import FastAPI
from loguru import logger

import src.classifier as clf
from routes.home import app_home
from routes.model_check import app_ridge_check
from routes.ridge_predict import app_ridge_predict

logger.add('logs/app_logs', level='INFO')
app = FastAPI(title="Ridge ML API", description="API Ridge model prediction", version="1.0")


@app.on_event('startup')
async def load_model():
    clf.model = pickle.load(open('models/ridge-regression-model.pkl', 'rb'))
    logger.info(f"Model Loaded! : {type(clf.model)}")
    # clf.model = load('models/ridge-regression-model.joblib')


app.include_router(app_home)
app.include_router(app_ridge_predict)
app.include_router(app_ridge_check)
