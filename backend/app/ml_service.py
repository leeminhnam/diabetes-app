import joblib
import json
import os
import pandas as pd
from .schemas import PredictRequest, PredictResponse

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models')

class DiabetesPredictor:
    def __init__(self):
        self.model = joblib.load(os.path.join(MODEL_DIR, 'best_diabetes_model.pkl'))
        self.scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_diabetes.pkl'))
        with open(os.path.join(MODEL_DIR, 'model_columns.json'), 'r', encoding='utf-8') as f:
            self.model_columns = json.load(f)

    def predict(self, data: PredictRequest) -> PredictResponse:
        try:
            # Convert to DataFrame
            df = pd.DataFrame([data.model_dump()])

            # One-hot encoding
            df = pd.get_dummies(df, columns=['gender', 'smoking_history'])

            # Align columns
            for col in self.model_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.model_columns]

            # Scale ALL features (because scaler was fit on all columns)
            scaled_array = self.scaler.transform(df)

            # Predict using the scaled array
            prediction = int(self.model.predict(scaled_array)[0])
            prob = self.model.predict_proba(scaled_array)[0][1]

            diagnosis = "Nguy cơ cao" if prediction == 1 else "Bình thường"

            return PredictResponse(
                success=True,
                prediction=prediction,
                risk_percentage=round(prob * 100, 1),
                diagnosis=diagnosis,
                model_used="XGBoost (Loaded Model)"
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

# Global instance
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        predictor = DiabetesPredictor()
    return predictor
