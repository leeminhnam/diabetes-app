from pydantic import BaseModel

class PredictRequest(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float

class PredictResponse(BaseModel):
    success: bool
    prediction: int
    risk_percentage: float
    diagnosis: str
    model_used: str
    error: str | None = None
