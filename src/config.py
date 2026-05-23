MODEL_PATH = "models/fraud_model.pkl"

DATA_PATH = "data/processed/transactions_processed.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = [
    "amount",
    "transaction_type",
    "country",
    "device_type",
    "time_since_last_tx"
]

