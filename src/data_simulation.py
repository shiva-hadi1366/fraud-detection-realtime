import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Output path
OUTPUT_PATH = "data/raw/transactions.csv"

# Reproducibility
np.random.seed(42)


def simulate_transactions(n=50000, fraud_rate=0.03):
    """
    Simuliert realitätsnahe Banktransaktionen inklusive Fraud-Label.
    """

    # Static lists
    countries = ["DE", "FR", "IT", "ES", "NL", "PL", "US", "CN"]
    devices = ["Mobile", "Web", "ATM", "POS"]
    merchants = ["Electronics", "Clothing", "Food", "Travel", "Crypto", "Gaming"]

    # Generate user IDs
    user_ids = np.random.randint(1000, 9999, size=n)

    # Generate device IDs
    device_ids = np.random.randint(10000, 99999, size=n)

    # Start timestamp
    start_time = datetime(2024, 1, 1)
    timestamps = []
    current_time = start_time

    for _ in range(n):
        delta = np.random.exponential(scale=60)
        current_time += timedelta(seconds=delta)
        timestamps.append(current_time)

    # Base dataset
    data = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "user_id": user_ids,
        "device_id": device_ids,
        "amount": np.round(np.random.exponential(scale=80, size=n), 2),
        "country": np.random.choice(countries, size=n),
        "device": np.random.choice(devices, size=n),
        "merchant": np.random.choice(merchants, size=n),
        "timestamp": timestamps,
    })

    # Time since last transaction
    data["time_since_last"] = data["timestamp"].diff().dt.total_seconds().fillna(0)

    # IP risk score (0–1)
    data["ip_risk_score"] = np.round(np.random.beta(2, 8, size=n), 3)

    # Fraud probability model
    fraud_prob = (
        fraud_rate
        + 0.03 * (data["amount"] > 500)
        + 0.04 * (data["country"].isin(["US", "CN"]))
        + 0.05 * (data["device"] == "Web")
        + 0.06 * (data["merchant"] == "Crypto")
        + 0.03 * (data["time_since_last"] < 5)
        + 0.10 * (data["ip_risk_score"] > 0.7)
    )

    # Rapid-fire fraud scenario
    fraud_prob += 0.12 * (data["time_since_last"] < 2)

    # Clip probabilities
    fraud_prob = fraud_prob.clip(0, 0.95)

    # Generate fraud labels
    data["is_fraud"] = np.random.binomial(1, fraud_prob)

    return data


def save_data(df):
    """Speichert das simulierte Dataset als CSV."""
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset erfolgreich gespeichert unter: {OUTPUT_PATH}")


if __name__ == "__main__":
    print("Simuliere Transaktionen...")
    df = simulate_transactions()
    save_data(df)
    print("Fertig!")
