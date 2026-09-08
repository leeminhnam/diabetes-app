import os
import joblib
import json
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class XGBoostService:
    def __init__(self):
        self.model = joblib.load(os.path.join(CURRENT_DIR, 'best_diabetes_model.pkl'))
        self.scaler = joblib.load(os.path.join(CURRENT_DIR, 'scaler_diabetes.pkl'))
        with open(os.path.join(CURRENT_DIR, 'model_columns.json'), 'r', encoding='utf-8') as f:
            self.model_columns = json.load(f)

    def predict(self, input_data: dict) -> dict:
        # Chuyển đổi dữ liệu input sang DataFrame
        df = pd.DataFrame([input_data])

        # Áp dụng One-Hot Encoding giống hệt quá trình huấn luyện
        df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

        # Căn chỉnh đầy đủ các cột (thiếu cột nào tự động bù 0)
        input_processed = df.reindex(columns=self.model_columns, fill_value=0)

        # Ép kiểu về float
        input_processed = input_processed.astype(float)

        # Chuẩn hóa
        scaled_array = self.scaler.transform(input_processed)

        # Thực hiện dự đoán
        prediction = int(self.model.predict(scaled_array)[0])
        prob = float(self.model.predict_proba(scaled_array)[0][1])

        return {
            "prediction": prediction,
            "risk_percentage": round(prob * 100, 1),
            "model_used": "Gradient Boosting (XGBoost)"
        }

# Global instance
xgboost_service = None

def get_xgboost_service():
    global xgboost_service
    if xgboost_service is None:
        xgboost_service = XGBoostService()
    return xgboost_service

