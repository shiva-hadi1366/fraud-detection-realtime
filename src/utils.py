import joblib
import pandas as pd

# === Imports required for loading the model ===
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def load_model(path):
    return joblib.load(path)

def save_model(model, path):
    joblib.dump(model, path)

def json_to_df(json_data):
    return pd.DataFrame([json_data])
