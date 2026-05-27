# 🚀 Fraud Detection System - Setup & Run Guide

## 📋 Project Overview

This is a complete end-to-end real-time fraud detection system with:
- ✅ **Data Simulation**: Generate realistic transaction data
- ✅ **ML Training**: Train XGBoost model with SMOTE for class imbalance
- ✅ **FastAPI**: REST API for real-time predictions
- ✅ **Streamlit Dashboard**: Interactive web interface
- ✅ **Batch Processing**: Analyze multiple transactions

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/shiva-hadi1366/fraud-detection-realtime.git
cd fraud-detection-realtime
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🎯 Getting Started - Step by Step

### Step 1: Generate Sample Data
```bash
python src/data_simulation.py
```
This creates `data/raw/transactions.csv` with 50,000 simulated transactions.

### Step 2: Train the Model
```bash
python src/train_models.py
```
This trains the XGBoost model and saves it to `models/fraud_model_advanced.pkl`.

**Output:**
- Model trained with SMOTE for imbalanced data
- Optimal threshold calculated
- Metrics displayed (Precision, Recall, F1-Score, AUC)

### Step 3: Run the FastAPI Server
```bash
python api/main.py
```
Or with uvicorn:
```bash
uvicorn api.main:app --reload --port 8000
```

**API Documentation:**
- 📖 Visit: http://localhost:8000/docs
- Test endpoint: `POST /predict`

### Step 4: Run the Streamlit Dashboard
In a new terminal:
```bash
streamlit run dashboard/app.py
```

**Dashboard Features:**
- 📊 Overview metrics and charts
- 📈 Advanced analytics by country/device
- 🔮 Single transaction prediction
- 📉 Batch analysis of multiple transactions

---

## 🔌 API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1234,
    "device_id": 56789,
    "amount": 150.50,
    "country": "DE",
    "device": "Mobile",
    "merchant": "Electronics",
    "time_since_last": 3600.0,
    "ip_risk_score": 0.25,
    "hour": 14,
    "day": 15,
    "weekday": 2
  }'
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/batch-predict \
  -H "Content-Type: application/json" \
  -d '[
    {"user_id": 1234, "device_id": 56789, ...},
    {"user_id": 5678, "device_id": 12345, ...}
  ]'
```

---

## 📂 Project Structure

```
fraud-detection-realtime/
├── data/
│   ├── raw/                    # Original transaction data
│   ├── processed/              # Processed data (if used)
├── src/
│   ├── config.py              # Configuration & paths
│   ├── data_simulation.py      # Generate synthetic transactions
│   ├── train_models.py         # Train XGBoost model
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── inference.py            # Model inference class
│   └── utils.py                # Utility functions
├── api/
│   └── main.py                 # FastAPI server
├── dashboard/
│   └── app.py                  # Streamlit interface
├── models/
│   └── fraud_model_advanced.pkl # Trained model
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
# Model paths
MODEL_PATH = "models/fraud_model_advanced.pkl"
RAW_DATA_PATH = "data/raw/transactions.csv"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Features
FEATURES = ["user_id", "device_id", "amount", "country", ...]
```

---

## 📊 Model Details

### Training Parameters
- **Algorithm**: XGBoost (n_estimators=400)
- **Class Imbalance**: SMOTE oversampling
- **Encoding**: One-Hot Encoding for categorical variables
- **Threshold**: Optimized for F1-Score

### Features Used
- `user_id`, `device_id`: User & device identifiers
- `amount`: Transaction amount
- `country`, `device`, `merchant`: Categorical features
- `time_since_last`: Time gap between transactions
- `ip_risk_score`: IP risk assessment
- `hour`, `day`, `weekday`: Temporal features

---

## 🔍 Fraud Detection Rules

The model considers multiple signals:
- ✅ High transaction amount
- ✅ Risky countries (US, CN)
- ✅ Web-based transactions
- ✅ Crypto merchant category
- ✅ Rapid-fire transactions (< 2 seconds apart)
- ✅ High IP risk score

---

## 🚨 Troubleshooting

### Model not loading
```
ERROR: Model not found at models/fraud_model_advanced.pkl
FIX: Run python src/train_models.py first
```

### Data not found
```
ERROR: Data file not found at data/raw/transactions.csv
FIX: Run python src/data_simulation.py first
```

### Import errors
```
ERROR: ModuleNotFoundError: No module named 'xgboost'
FIX: pip install -r requirements.txt
```

### Port already in use
```
ERROR: Address already in use (port 8000)
FIX: uvicorn api.main:app --port 8001
```

---

## 🎨 Dashboard Pages

### 🏠 Dashboard
- Total transactions count
- Fraudulent cases statistics
- Transaction amount distribution
- Real-time metrics

### 📊 Analytics
- Fraud distribution by country
- Fraud distribution by device type
- Amount analysis by fraud status
- Detailed charts and visualizations

### 🔮 Make Prediction
- Input transaction details
- Real-time fraud prediction
- Risk level indicator (Low/Medium/High)
- Detailed transaction breakdown

### 📈 Batch Analysis
- Analyze multiple transactions
- Compare predicted vs actual labels
- Summary statistics
- Progress tracking

---

## 📈 Performance Metrics

Expected model performance:
- **Precision**: ~75-85% (minimize false positives)
- **Recall**: ~70-80% (catch most fraudulent transactions)
- **F1-Score**: ~75% (balanced performance)
- **AUC-ROC**: ~0.85+ (good discrimination)

---

## 🔐 Security Notes

- Models are stored in `.gitkeep` directories by default
- Never commit sensitive data or model weights
- Use environment variables for API keys
- Validate all input data before prediction

---

## 📝 Next Steps

### Enhancements
- [ ] Add database integration (PostgreSQL)
- [ ] Implement model versioning
- [ ] Add explainability (SHAP values)
- [ ] Deploy with Docker
- [ ] Setup CI/CD pipeline
- [ ] Add real-time Kafka streaming
- [ ] Implement A/B testing

### Production Deployment
```bash
# Using Docker
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection

# Using AWS/Azure/GCP
# Configure environment and deploy
```

---

## 📧 Contact & Support

For issues or questions:
- GitHub: https://github.com/shiva-hadi1366
- Create an issue in the repository

---

## 📄 License

This project is open source. Feel free to use and modify!

---

**Happy Fraud Detection! 🎯**
