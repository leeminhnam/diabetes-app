from fastapi import APIRouter
from .schemas import PredictRequest, PredictResponse
from .ml_service import get_predictor

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict_diabetes(request: PredictRequest):
    predictor = get_predictor()
    return predictor.predict(request)
