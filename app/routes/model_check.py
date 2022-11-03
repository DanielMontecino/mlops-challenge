import numpy as np
import src.classifier as clf
from fastapi import APIRouter
from loguru import logger
from src.config import N_FEATURES
from src.ridge_classifier import RidgeCheckResponse

app_ridge_check = APIRouter()


@app_ridge_check.post('/ridge/check',
                      tags=["Check"],
                      response_model=RidgeCheckResponse,
                      description="Get Regression Response from random data")
async def check_model():
    logger.info("Check model")
    data = np.random.rand(1, N_FEATURES)
    logger.info(f"Data: {data}")
    logger.info(f"Model: {type(clf)}")
    prediction = clf.model.predict(data).tolist()
    logger.info(f"Prediction: {prediction} [{type(prediction[0][0])}]")
    return {"data": data.tolist(),
            "prediction": prediction}
