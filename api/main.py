"""
FastAPI server for real-time fraud detection predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FraudInference
from src.utils import load_model
from src.config import MODEL_PATH

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for financial transactions",
    version="1.0.0"
)

# Initialize model inference
try:
    inference = FraudInference()
except Exception as e:
    print(f"Warning: Could not load model at startup: {e}")
    inference = None


# === Pydantic Models ===
class TransactionInput(BaseModel):
    """Input schema for transaction data"""
    user_id: int
    device_id: int
    amount: float
    country: str
    device: str
    merchant: str
    time_since_last: float
    ip_risk_score: float
    hour: int
    day: int
    weekday: int

    class Config:
        schema_extra = {
            "example": {
                "user_id": 1234,
                "device_id": 56789,
                "amount": 150.50,
                "country": "DE",
                "device": "Mobile",
                "merchant": "Electronics",
                "time_since_last": 3600.0,
                "ip_risk_score": 0.25,
                "hour": 14,
                "day": 15,
                "weekday": 2
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for fraud prediction"""
    fraud_probability: float
    fraud_label: int
    risk_level: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool


# === API Endpoints ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy" if inference else "model_not_loaded",
        "model_loaded": inference is not None
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_fraud(transaction: TransactionInput):
    """
    Predict if a transaction is fraudulent
    
    Returns:
    - fraud_probability: Confidence score (0-1)
    - fraud_label: 0 (legitimate) or 1 (fraudulent)
    - risk_level: "low", "medium", "high"
    """
    
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to dict
        data = transaction.dict()
        
        # Get prediction
        result = inference.predict(data)
        
        # Determine risk level
        prob = result["fraud_probability"]
        if prob < 0.3:
            risk_level = "low"
        elif prob < 0.7:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        from datetime import datetime
        
        return {
            "fraud_probability": result["fraud_probability"],
            "fraud_label": result["fraud_label"],
            "risk_level": risk_level,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(transactions: list[TransactionInput]):
    """
    Predict fraud for multiple transactions
    """
    
    if inference is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for transaction in transactions:
            data = transaction.dict()
            prediction = inference.predict(data)
            
            prob = prediction["fraud_probability"]
            if prob < 0.3:
                risk_level = "low"
            elif prob < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            results.append({
                "fraud_probability": prediction["fraud_probability"],
                "fraud_label": prediction["fraud_label"],
                "risk_level": risk_level
            })
        
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/")
async def root():
    """Welcome message"""
    return {
        "message": "Welcome to Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
