from fastapi import APIRouter
from .schemas import PredictRequest, PredictResponse
from .xgboost.service import get_xgboost_service
from .mlp.service import get_mlp_service

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict_diabetes(request: PredictRequest):
    input_data = request.model_dump()
    model_type = input_data.pop('model_type', 'xgboost')

    try:
        if model_type == 'mlp':
            service = get_mlp_service()
            result = service.predict(input_data)
        else:
            service = get_xgboost_service()
            result = service.predict(input_data)

        diagnosis = "Nguy cơ cao mắc tiểu đường (Dương tính)" if result["prediction"] == 1 else "Bình thường / Ít nguy cơ (Âm tính)"

        return PredictResponse(
            success=True,
            prediction=result["prediction"],
            risk_percentage=result["risk_percentage"],
            diagnosis=diagnosis,
            model_used=result["model_used"]
        )
    except Exception as e:
        return PredictResponse(
            success=False,
            prediction=0,
            risk_percentage=0.0,
            diagnosis="Error",
            model_used="",
            error=str(e)
        )

