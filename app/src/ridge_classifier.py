from typing import List

from pydantic import BaseModel, conlist
from src.config import N_FEATURES


class Ridge(BaseModel):
    data: List[conlist(float, min_items=N_FEATURES, max_items=N_FEATURES)]


class RidgePredictionResponse(BaseModel):
    prediction: List[conlist(float)]


class RidgeCheckResponse(BaseModel):
    data: List[conlist(float, min_items=N_FEATURES, max_items=N_FEATURES)]
    prediction: List[conlist(float)]
