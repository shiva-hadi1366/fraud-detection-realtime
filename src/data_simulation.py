import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# Pfad für die Ausgabe
OUTPUT_PATH = "data/raw/transactions.csv"

# Zufälligen Seed setzen (Reproduzierbarkeit)
np.random.seed(42)

def simulate_transactions(n=50000):
    """
    Simuliert realitätsnahe Banktransaktionen inklusive Fraud-Label.
    """

    # Länder (vereinfachte Auswahl)
    countries = ["DE", "FR", "IT", "ES", "NL", "PL", "US", "CN"]

    # Gerätetypen
    devices = ["Mobile", "Web", "ATM", "POS"]

    # Startzeitpunkt
    start_time = datetime(2024, 1, 1)

    timestamps = []
    current_time = start_time

    for _ in range(n):
        # Zeit zwischen Transaktionen (0–300 Sekunden)
        delta = np.random.exponential(scale=60)
        current_time += timedelta(seconds=delta)
        timestamps.append(current_time)

    # Features generieren
    data = pd.DataFrame({
        "transaction_id": np.arange(1, n + 1),
        "amount": np.round(np.random.exponential(scale=80, size=n), 2),  # typische Beträge
        "country": np.random.choice(countries, size=n),
        "device": np.random.choice(devices, size=n),
        "timestamp": timestamps,
    })

    # Zeit seit letzter Transaktion
    data["time_since_last"] = data["timestamp"].diff().dt.total_seconds().fillna(0)

    # Fraud-Wahrscheinlichkeiten basierend auf Mustern
    fraud_prob = (
        0.02
        + 0.03 * (data["amount"] > 500)
        + 0.04 * (data["country"].isin(["US", "CN"]))
        + 0.05 * (data["device"] == "Web")
        + 0.03 * (data["time_since_last"] < 5)
    )

    fraud_prob = fraud_prob.clip(0, 0.9)

    # Fraud-Label generieren
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

