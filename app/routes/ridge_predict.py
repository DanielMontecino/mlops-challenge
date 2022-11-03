import src.classifier as clf
from fastapi import APIRouter
from loguru import logger
from src.ridge_classifier import Ridge, RidgePredictionResponse

app_ridge_predict = APIRouter()


@app_ridge_predict.post('/ridge/predict',
                        tags=["Predictions"],
                        response_model=RidgePredictionResponse,
                        description="Get Regression Response")
async def get_prediction(ridge: Ridge):
    data = dict(ridge)['data']
    logger.info(f"Data: {data}")
    logger.info(f"Model: {type(clf)}")
    prediction = clf.model.predict(data).tolist()
    logger.info(f"Prediction: {prediction} [{type(prediction[0][0])}]")
    return {"prediction": prediction}
