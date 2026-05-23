import pickle
import pandas as pd

def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def json_to_df(json_data):
    return pd.DataFrame([json_data])

