import os

# --- Base directory of the project ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# --- Paths ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "artifacts", "fraud_model_advanced.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "transactions_processed.csv")

# --- Training parameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Features used for training & inference ---
FEATURES = [
    "amount",
    "transaction_type",
    "country",
    "device_type",
    "time_since_last_tx"
]
