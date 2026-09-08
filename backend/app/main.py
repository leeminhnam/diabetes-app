from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router
from .xgboost.service import get_xgboost_service
from .mlp.service import get_mlp_service

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
    # Load models into memory on startup
    get_xgboost_service()
    get_mlp_service()

app.include_router(router)

