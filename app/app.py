import os
import pickle

from fastapi import FastAPI
from loguru import logger

import src.classifier as clf
from routes.home import app_home
from routes.model_check import app_ridge_check
from routes.ridge_predict import app_ridge_predict
from src.config import MODELS_PATH, MODEL_PKL

logger.add('logs/app_logs', level='INFO')
app = FastAPI(title="Ridge ML API", description="API Ridge model prediction", version="1.0")


@app.on_event('startup')
async def load_model():
    clf.model = pickle.load(open(os.path.join(MODELS_PATH, MODEL_PKL), 'rb'))
    logger.info(f"Model Loaded! : {type(clf.model)}")


app.include_router(app_home)
app.include_router(app_ridge_predict)
app.include_router(app_ridge_check)
