MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/scaler.pkl"

TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURES = [
    "amount",
    "transaction_type",
    "country",
    "device_type",
    "time_since_last_tx"
]

