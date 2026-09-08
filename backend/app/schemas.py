from pydantic import BaseModel, ConfigDict

class PredictRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    model_type: str = "xgboost"

class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    success: bool
    prediction: int
    risk_percentage: float
    diagnosis: str
    model_used: str
    error: str | None = None
