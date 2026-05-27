import os

# --- Base directory of the project ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --- Paths ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_model_advanced.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_processed.csv")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "transactions.csv")

# --- Training parameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Features used for training & inference ---
FEATURES = [
    "user_id",
    "device_id",
    "amount",
    "country",
    "device",
    "merchant",
    "time_since_last",
    "ip_risk_score",
    "hour",
    "day",
    "weekday"
]
