# FastAPI Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Diabetes Prediction backend using FastAPI to serve the existing machine learning model to the React frontend.

**Architecture:** A modular FastAPI application with separate files for routing (`api.py`), validation (`schemas.py`), and machine learning logic (`ml_service.py`). The frontend will be updated to point to the new API.

**Tech Stack:** Python, FastAPI, Uvicorn, Pydantic, Scikit-learn, Pandas, React.

**Spec:** `docs/superpowers/specs/2026-09-08-fastapi-backend-design.md`

## Global Constraints

- Backend must run on `http://127.0.0.1:8000`.
- API endpoint must be `POST /predict`.
- Must load `.pkl` models once at startup.
- Frontend must be updated to use the new backend URL.

---

### Task 1: Setup Backend Environment and Directory Structure

**Files:**
- Create: `backend/requirements.txt`
- Modify: (Move model files from `backend/` to `backend/ml_models/`)

**Interfaces:**
- Consumes: None
- Produces: Project environment and file structure ready.

- [ ] **Step 1: Write requirements.txt**
Create `backend/requirements.txt` with necessary dependencies.
```text
fastapi==0.110.0
uvicorn==0.27.1
pydantic==2.6.3
scikit-learn==1.4.1.post1
pandas==2.2.1
xgboost==2.0.3
```

- [ ] **Step 2: Reorganize directories**
Move the model files to `backend/ml_models/`.
```bash
mkdir -p backend/app backend/ml_models
mv backend/best_diabetes_model.pkl backend/ml_models/
mv backend/scaler_diabetes.pkl backend/ml_models/
mv backend/model_columns.json backend/ml_models/
```

- [ ] **Step 3: Commit**
```bash
git add backend/requirements.txt backend/ml_models/
git commit -m "chore: setup backend directories and requirements"
```

---

### Task 2: Implement Data Validation (schemas.py)

**Files:**
- Create: `backend/app/schemas.py`

**Interfaces:**
- Consumes: None
- Produces: `PredictRequest` and `PredictResponse` Pydantic models.

- [ ] **Step 1: Write schemas.py**
```python
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
```

- [ ] **Step 2: Commit**
```bash
git add backend/app/schemas.py
git commit -m "feat: add pydantic schemas for request and response"
```

---

### Task 3: Implement ML Service (ml_service.py)

**Files:**
- Create: `backend/app/ml_service.py`

**Interfaces:**
- Consumes: `PredictRequest` from `backend/app/schemas.py`, `.pkl` and `.json` files.
- Produces: `DiabetesPredictor` class with `predict(data: PredictRequest)` method returning `PredictResponse`.

- [ ] **Step 1: Write ml_service.py**
```python
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
```

- [ ] **Step 2: Commit**
```bash
git add backend/app/ml_service.py
git commit -m "feat: implement machine learning service"
```

---

### Task 4: Implement API Routing and Main App (api.py & main.py)

**Files:**
- Create: `backend/app/api.py`, `backend/app/main.py`, `backend/app/__init__.py`

**Interfaces:**
- Consumes: `PredictRequest`, `PredictResponse`, `get_predictor`.
- Produces: FastAPI application running on port 8000.

- [ ] **Step 1: Write api.py**
```python
from fastapi import APIRouter
from .schemas import PredictRequest, PredictResponse
from .ml_service import get_predictor

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict_diabetes(request: PredictRequest):
    predictor = get_predictor()
    return predictor.predict(request)
```

- [ ] **Step 2: Write main.py**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router
from .ml_service import get_predictor

app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For dev purposes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    get_predictor() # Load models

app.include_router(router)
```

- [ ] **Step 3: Create __init__.py**
```bash
touch backend/app/__init__.py
```

- [ ] **Step 4: Commit**
```bash
git add backend/app/api.py backend/app/main.py backend/app/__init__.py
git commit -m "feat: add fastapi routing and main app configuration"
```

---

### Task 5: Update Frontend Connection

**Files:**
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: FastAPI server on `http://127.0.0.1:8000/predict`.

- [ ] **Step 1: Modify frontend/src/App.jsx**
Change the fetch URL from `http://127.0.0.1:5000/predict` to `http://127.0.0.1:8000/predict` and update error message.
Run: `sed -i 's/5000/8000/g' frontend/src/App.jsx`
*(Note: If sed doesn't work well on Windows PowerShell, we will replace the string in the file manually or using Python script during execution)*

- [ ] **Step 2: Commit**
```bash
git add frontend/src/App.jsx
git commit -m "fix: update api endpoint to port 8000 for fastapi"
```
