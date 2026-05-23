import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ============================
# Path Setup
# ============================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model_advanced.pkl")

# ============================
# Load Dataset
# ============================

df = pd.read_csv(os.path.join(BASE_DIR, "data/raw/transactions.csv"))

df = df.drop(columns=["transaction_id"])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.day
df["weekday"] = df["timestamp"].dt.weekday
df = df.drop(columns=["timestamp"])

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

categorical = ["country", "device"]
numerical = [col for col in X.columns if col not in categorical]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numerical)
    ]
)

model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=5,
    random_state=42,
    n_jobs=-1
)

pipeline = ImbPipeline(steps=[
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("model", model)
])

# ============================
# Train-Test Split
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_proba = pipeline.predict_proba(X_test)[:, 1]

# ============================
# Best Threshold (F1)
# ============================

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best threshold: {best_threshold:.3f}")
print(f"Precision: {precisions[best_idx]:.3f}")
print(f"Recall: {recalls[best_idx]:.3f}")
print(f"Best F1: {f1_scores[best_idx]:.3f}")

# Default threshold evaluation
y_pred_default = (y_proba >= 0.5).astype(int)
print("\n=== Metrics at threshold 0.5 ===")
print(classification_report(y_test, y_pred_default))
print("AUC:", roc_auc_score(y_test, y_proba))

# Best threshold evaluation
y_pred_best = (y_proba >= best_threshold).astype(int)
print("\n=== Metrics at best threshold ===")
print(classification_report(y_test, y_pred_best))

# ============================
# Save Model + Threshold
# ============================

artifact = {
    "model": pipeline,
    "threshold": float(best_threshold)
}

joblib.dump(artifact, MODEL_PATH)
print(f"\nModel saved successfully at:\n{MODEL_PATH}")
