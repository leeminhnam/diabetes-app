import pickle
import json
import os
import pandas as pd
from .schemas import PredictRequest, PredictResponse

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_models')

class DiabetesPredictor:
    def __init__(self):
        with open(os.path.join(MODEL_DIR, 'best_diabetes_model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'scaler_diabetes.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'model_columns.json'), 'r') as f:
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

            # Scale numerical features
            num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
            df[num_cols] = self.scaler.transform(df[num_cols])

            # Predict
            prediction = int(self.model.predict(df)[0])
            prob = self.model.predict_proba(df)[0][1]

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
