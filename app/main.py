import json
import math
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

class PatientFeatures(BaseModel):
    Pregnancies: int = Field(..., ge=0)
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int = Field(..., ge=0)

class PredictResponse(BaseModel):
    prediction: int
    result: str
    confidence: float
    prob_positive: float
    threshold: float

app = FastAPI(title="Diabetes Prediction API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


ARTIFACT_PATH = Path("models/diabetes_model.pkl")
ARTIFACT: Optional[Dict[str, Any]] = None
MODEL = None
THRESHOLD: float = 0.5
FEATURES: list[str] = []
METRICS: Dict[str, Any] = {}


@app.on_event("startup")
async def _load_artifact():
    global ARTIFACT, MODEL, THRESHOLD, FEATURES, METRICS

    if not ARTIFACT_PATH.exists():
        raise RuntimeError(
            f"Artifact not found at {ARTIFACT_PATH}. Train first: `python train.py`"
        )

    ARTIFACT = joblib.load(ARTIFACT_PATH)

    MODEL = ARTIFACT.get("model")
    THRESHOLD = float(ARTIFACT.get("threshold", 0.5))
    FEATURES = ARTIFACT.get("feature_order", [])
    METRICS = ARTIFACT.get("metrics", {})

    if MODEL is None or not FEATURES:
        raise RuntimeError("Invalid artifact: missing model or features")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    if not METRICS:
        raise HTTPException(status_code=404, detail="No metrics available")
    return METRICS


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PatientFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = payload.model_dump()
    try:
        x = pd.DataFrame([[data[f] for f in FEATURES]], columns=FEATURES)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature: {e}")

    prob_pos = await run_in_threadpool(
        lambda: float(MODEL.predict_proba(x)[0, 1])
    )
    label = int(prob_pos >= THRESHOLD)
    result = "Diabetic" if label == 1 else "Not Diabetic"
    confidence = prob_pos if label == 1 else (1.0 - prob_pos)

    return PredictResponse(
        prediction=label,
        result=result,
        confidence=round(confidence, 4),
        prob_positive=round(prob_pos, 4),
        threshold=round(THRESHOLD, 4),
    )
