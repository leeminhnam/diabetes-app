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
