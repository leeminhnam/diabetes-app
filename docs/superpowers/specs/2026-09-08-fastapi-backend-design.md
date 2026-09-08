# Backend Architecture Design: Diabetes Prediction FastAPI

## 1. Context and Goals
The goal is to rebuild the backend for the Diabetes Prediction application using **FastAPI** (Python). The existing implementation used Flask (or a Jupyter notebook directly). The new backend must serve the pre-trained machine learning model (`best_diabetes_model.pkl`) to the existing React frontend, maintaining high performance, clean architecture, and strict input validation.

## 2. Directory Structure
We adopt a modular structure to separate concerns (routing, validation, ML logic).

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entry point, CORS middleware
│   ├── api.py           # API routing (e.g., POST /predict)
│   ├── schemas.py       # Pydantic schemas for request and response validation
│   └── ml_service.py    # Machine learning logic (model loading, preprocessing, inference)
├── ml_models/           # Extracted from backend root
│   ├── best_diabetes_model.pkl
│   ├── scaler_diabetes.pkl
│   └── model_columns.json
└── requirements.txt     # Python dependencies
```

## 3. Data Flow & Components

### 3.1. Data Validation (schemas.py)
Uses **Pydantic** to validate incoming JSON payloads. 
**Input fields:**
- `gender` (str): 'Female' or 'Male'
- `age` (float): Patient's age
- `hypertension` (int): 0 or 1
- `heart_disease` (int): 0 or 1
- `smoking_history` (str): e.g., 'never', 'current', 'former'
- `bmi` (float): Body Mass Index
- `HbA1c_level` (float): Hemoglobin A1c level
- `blood_glucose_level` (float): Blood glucose level

### 3.2. ML Service (ml_service.py)
- **Initialization**: Loads `best_diabetes_model.pkl`, `scaler_diabetes.pkl`, and `model_columns.json` into memory globally when the application starts.
- **Preprocessing**: 
  - Converts Pydantic input to a Pandas DataFrame.
  - Applies One-Hot Encoding to categorical variables (`gender`, `smoking_history`).
  - Aligns columns with `model_columns.json` (fills missing columns with 0).
  - Scales numerical features using the loaded `StandardScaler`.
- **Inference**: Calls `model.predict()` and `model.predict_proba()` to determine the class (0 or 1) and calculate the risk percentage.

### 3.3. API Endpoint (api.py & main.py)
- **Endpoint**: `POST /predict`
- **Output Schema**:
  ```json
  {
    "success": true,
    "prediction": 1,
    "risk_percentage": 85.5,
    "diagnosis": "Nguy cơ cao",
    "model_used": "XGBoost (or actual model name)"
  }
  ```
- **CORS**: Configured in `main.py` to allow requests from `http://localhost:5173` and `http://127.0.0.1:5173`.

## 4. Frontend Integration
- Modify `frontend/src/App.jsx` to point to `http://127.0.0.1:8000/predict` instead of the old `5000` port.

## 5. Deployment / Running Instructions
1. Install dependencies: `pip install -r backend/requirements.txt`
2. Run backend: `cd backend && uvicorn app.main:app --reload`
3. Run frontend: `cd frontend && npm run dev`
