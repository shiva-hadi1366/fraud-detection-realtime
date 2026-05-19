📘 Fraud Detection Realtime 
# 🔍 Real-Time Fraud Detection System  
Ein End-to-End Machine-Learning-Projekt zur Erkennung von betrügerischen Transaktionen in Echtzeit – inklusive Datenanalyse, Modelltraining, FastAPI-Backend und Streamlit-Dashboard.

---

## 🚀 Projektüberblick

Dieses Projekt implementiert ein vollständiges Fraud-Detection-System, das:

- Transaktionsdaten analysiert  
- Ein ML-Modell trainiert  
- Betrug in Echtzeit über eine API vorhersagt  
- Ein interaktives Dashboard zur Visualisierung bereitstellt  

Das Ziel ist es, ein skalierbares, produktionsreifes System zu entwickeln, das in realen Finanzumgebungen eingesetzt werden kann.

---

## 📂 Projektstruktur

fraud-detection-realtime/
│
├── data/
│   ├── raw/                # Originaldaten
│   ├── processed/          # Bereinigte Daten
│
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── model_training.ipynb
│
├── src/
│   ├── data_prep.py        # Datenbereinigung
│   ├── features.py         # Feature Engineering
│   ├── train.py            # Modelltraining
│   ├── evaluate.py         # Modellbewertung
│
├── api/
│   ├── main.py             # FastAPI Endpoint
│
├── dashboard/
│   ├── app.py              # Streamlit Dashboard
│
├── models/
│   ├── fraud_model.pkl     # Trainiertes Modell
│
├── README.md
└── requirements.txt

Code

---

# 📊 **Exploratory Data Analysis (EDA)**

Eine umfassende Analyse der Transaktionsdaten, um Muster, Ausreißer und potenzielle Fraud-Indikatoren zu identifizieren.

---

## 📥 1. Laden des Datensatzes

```python
df = pd.read_csv("data/raw/transactions.csv")
df.head()
🧾 2. Überblick über Struktur & Datenqualität
python
df.info()
df.describe()
df.isnull().sum()
Erkenntnisse
Keine fehlenden Werte

Mischung aus numerischen und kategorialen Variablen

Zielvariable: is_fraud

📊 3. Verteilungen der numerischen Variablen
python
df.hist(figsize=(15, 10), bins=30)
plt.show()
Erkenntnisse
amount und time_since_last sind rechtsschief

transaction_id ist gleichmäßig verteilt

is_fraud zeigt starke Klassen-Imbalance

⚖️ 4. Verteilung der Zielvariable
python
sns.countplot(data=df, x="is_fraud")
plt.title("Fraud vs Non-Fraud Distribution")
Erkenntnisse
Fraud-Fälle sind selten

Wichtig für Modellierung (Class Weights, Oversampling)

🔥 5. Korrelationsanalyse
python
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), cmap="coolwarm", annot=True)
Erkenntnisse
Sehr geringe Korrelationen

Typisch für Fraud Detection

Feature Engineering wird entscheidend

📦 6. Betragsanalyse nach Fraud-Status
python
sns.boxplot(data=df, x="is_fraud", y="amount")
Erkenntnisse
Median ähnlich

Viele Ausreißer

Fraud zeigt leicht auffällige Extremwerte

🤖 Modellierung
Das ML-Modell umfasst:

Datenbereinigung

Feature Engineering

Train/Test Split

Training eines Klassifikationsmodells (z. B. Random Forest, XGBoost)

Evaluierung mit:

Precision

Recall

F1-Score

ROC-AUC

🧪 Modellbewertung
Typische Metriken:

Recall ist besonders wichtig (Fraud nicht verpassen)

Precision ebenfalls relevant (False Positives reduzieren)

🌐 API (FastAPI)
Das trainierte Modell wird über eine REST-API bereitgestellt:

bash
uvicorn api.main:app --reload
Endpunkt:
Code
POST /predict
Input: JSON mit Transaktionsdaten
Output: Fraud = 0/1 + Wahrscheinlichkeit

📊 Dashboard (Streamlit)
Starten:

bash
streamlit run dashboard/app.py
Features:

Live-Vorhersagen

Visualisierung der Transaktionsverteilung

Fraud-Statistiken

🛠️ Installation
bash
pip install -r requirements.txt
🚀 Nächste Schritte
Erweiterte Feature Engineering

Hyperparameter Tuning

Deployment in Docker

Kafka für echte Echtzeit-Streams

CI/CD Pipeline

👤 Autor
Mohammadhadi Shiva  
Data Science Trainee – Deutschland
GitHub: https://github.com/shiva-hadi1366 (github.com in Bing)
