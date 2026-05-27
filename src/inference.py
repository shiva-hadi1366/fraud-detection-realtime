import pandas as pd
import os
import sys

# Fix relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import load_model
from config import MODEL_PATH, FEATURES

class FraudInference:
    def __init__(self):
        artifact = load_model(MODEL_PATH)
        self.model = artifact["model"]
        self.threshold = artifact["threshold"]

    def preprocess_input(self, data: dict) -> pd.DataFrame:
        df = pd.DataFrame([data])

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df = df[FEATURES]
        return df

    def predict(self, data: dict) -> dict:
        X = self.preprocess_input(data)
        prob = self.model.predict_proba(X)[0][1]
        label = int(prob >= self.threshold)

        return {
            "fraud_probability": float(prob),
            "fraud_label": label
        }
