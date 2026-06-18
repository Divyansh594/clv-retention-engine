from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="CLV + Retention Engine API",
    description="Predict Customer Lifetime Value and Churn Probability",
    version="1.0.0"
)

# ── Load models at startup ──
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    clv_model   = joblib.load(f"{BASE}/models/clv_rf_model.pkl")
    clv_scaler  = joblib.load(f"{BASE}/models/clv_scaler.pkl")
    clv_features = joblib.load(f"{BASE}/models/clv_features.pkl")

    churn_model   = joblib.load(f"{BASE}/models/churn_rf_model.pkl")
    churn_scaler  = joblib.load(f"{BASE}/models/churn_scaler.pkl")
    churn_features = joblib.load(f"{BASE}/models/churn_features.pkl")
    MODELS_LOADED = True
except Exception as e:
    print(f"Warning: Could not load models — {e}")
    MODELS_LOADED = False


# ── Request Schemas ──
class CustomerFeatures(BaseModel):
    Recency: float
    Frequency: float
    Monetary: float
    AvgOrderValue: float
    StdOrderValue: float = 0.0
    TotalTransactions: float
    UniqueProducts: float
    TotalQuantity: float
    ActiveMonths: float
    LifespanDays: float
    FreqPerMonth: float
    AvgInterPurchaseDays: float = 30.0
    R_Score: int = 3
    F_Score: int = 3
    M_Score: int = 3
    RFM_Score: int = 9


class CLVResponse(BaseModel):
    customer_id: str
    predicted_clv: float
    clv_segment: str
    confidence: str


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk: str
    recommended_action: str


# ── Helpers ──
def clv_segment(value: float) -> str:
    if value > 1000: return "High Value"
    if value > 300:  return "Medium Value"
    return "Low Value"

def churn_risk_label(prob: float) -> str:
    if prob > 0.7:   return "High Risk"
    if prob > 0.4:   return "Medium Risk"
    return "Low Risk"

def churn_action(prob: float) -> str:
    if prob > 0.7:   return "Immediate win-back campaign with 20% discount"
    if prob > 0.4:   return "Send re-engagement email with 10% offer"
    return "Regular loyalty nurture — no urgency"


# ── Endpoints ──
@app.get("/")
def root():
    return {"status": "running", "models_loaded": MODELS_LOADED}


@app.post("/predict_clv", response_model=CLVResponse)
def predict_clv(data: CustomerFeatures, customer_id: str = "unknown"):
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    features_dict = {k: getattr(data, k) for k in clv_features}
    X = pd.DataFrame([features_dict])[clv_features].fillna(0)
    X_sc = clv_scaler.transform(X)
    predicted = float(clv_model.predict(X_sc)[0])
    predicted = max(0, round(predicted, 2))

    return CLVResponse(
        customer_id=customer_id,
        predicted_clv=predicted,
        clv_segment=clv_segment(predicted),
        confidence="High" if data.Frequency > 5 else "Medium"
    )


@app.post("/predict_churn", response_model=ChurnResponse)
def predict_churn(data: CustomerFeatures, customer_id: str = "unknown"):
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    features_dict = {k: getattr(data, k) for k in churn_features}
    X = pd.DataFrame([features_dict])[churn_features].fillna(0)
    X_sc = churn_scaler.transform(X)
    prob = float(churn_model.predict_proba(X_sc)[0][1])
    prob = round(prob, 4)

    return ChurnResponse(
        customer_id=customer_id,
        churn_probability=prob,
        churn_risk=churn_risk_label(prob),
        recommended_action=churn_action(prob)
    )


@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": MODELS_LOADED}


# Run with: uvicorn src.api.main:app --reload --port 8000