import os
import joblib
import json
import pandas as pd
import torch
import torch.nn.functional as F
from .model import DeeperMLP

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class MLPService:
    def __init__(self):
        # Dùng chung scaler và cấu trúc cột với XGBoost (sẽ được copy vào folder này)
        self.scaler = joblib.load(os.path.join(CURRENT_DIR, 'scaler_diabetes.pkl'))
        with open(os.path.join(CURRENT_DIR, 'model_columns.json'), 'r', encoding='utf-8') as f:
            self.model_columns = json.load(f)

        # Load PyTorch model
        # Đảm bảo file mlp_model.pth mới (có input_dim=13) đã được huấn luyện qua file train_mlp_13.py
        self.model = DeeperMLP(input_dim=len(self.model_columns), num_classes=2)
        try:
            self.model.load_state_dict(torch.load(os.path.join(CURRENT_DIR, 'mlp_model.pth'), map_location=torch.device('cpu'), weights_only=True))
        except RuntimeError:
            print("CẢNH BÁO: mlp_model.pth hiện tại không khớp với 13 features. Vui lòng chạy file train_mlp_13.py trước.")

        self.model.eval()

    def predict(self, input_data: dict) -> dict:
        # Chuyển đổi dữ liệu input sang DataFrame
        df = pd.DataFrame([input_data])

        # Áp dụng One-Hot Encoding giống hệt quá trình huấn luyện XGBoost
        df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)

        # Căn chỉnh đầy đủ các cột (thiếu cột nào tự động bù 0)
        input_processed = df.reindex(columns=self.model_columns, fill_value=0)

        # Ép kiểu về float
        input_processed = input_processed.astype(float)

        # Chuẩn hóa (Dùng chung Scaler của XGBoost)
        scaled_array = self.scaler.transform(input_processed)

        # Predict với PyTorch
        tensor_X = torch.tensor(scaled_array, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(tensor_X)
            probs = F.softmax(logits, dim=1)
            prediction = int(torch.argmax(probs, dim=1)[0])
            prob = probs[0][1].item()

        return {
            "prediction": prediction,
            "risk_percentage": round(prob * 100, 1),
            "model_used": "Deep Learning (PyTorch MLP)"
        }

# Global instance
mlp_service = None

def get_mlp_service():
    global mlp_service
    if mlp_service is None:
        mlp_service = MLPService()
    return mlp_service

