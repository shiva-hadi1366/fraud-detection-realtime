# 🔍 Real-Time Fraud Detection System

**End-to-End Machine Learning Project for Detecting Fraudulent Transactions in Real-Time**

A comprehensive fraud detection system combining data analysis, machine learning, FastAPI backend, and Streamlit dashboard to identify suspicious financial transactions with high accuracy and scalability.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=flat-square&logo=streamlit)](https://streamlit.io/)
[![ML](https://img.shields.io/badge/ML-XGBoost%20|%20Random%20Forest-yellow?style=flat-square)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

---

## 🎯 Project Overview

This project implements a **production-ready fraud detection system** that:

- ✅ **Analyzes transaction data** to identify patterns and anomalies
- ✅ **Trains ML models** (XGBoost, Random Forest) with class balancing
- ✅ **Predicts fraud in real-time** via REST API endpoints
- ✅ **Provides interactive dashboard** for monitoring and visualization
- ✅ **Achieves high recall & precision** to minimize losses and false alarms

**Target Use Case:** Financial institutions, payment processors, and fintech platforms

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Precision** | ~95% |
| **Recall** | ~92% |
| **F1-Score** | ~93% |
| **ROC-AUC** | ~98% |
| **Processing Speed** | <50ms per transaction |

---

## 📁 Project Structure

```
fraud-detection-realtime/
│
├── data/
│   ├── raw/                    # Original transaction datasets
│   └── processed/              # Cleaned & engineered data
│
├── notebooks/
│   ├── eda.ipynb              # Exploratory Data Analysis
│   ├── model_training.ipynb   # Model development & tuning
│   └── evaluation.ipynb       # Performance analysis
│
├── src/
│   ├── data_prep.py           # Data cleaning & preprocessing
│   ├── features.py            # Feature engineering pipeline
│   ├── train.py               # Model training & validation
│   ├── evaluate.py            # Evaluation metrics
│   └── utils.py               # Helper functions
│
├── api/
│   ├── main.py                # FastAPI application
│   └── models.py              # Request/response schemas
│
├── dashboard/
│   ├── app.py                 # Streamlit dashboard
│   └── pages/                 # Multi-page components
│
├── models/
│   └── fraud_model.pkl        # Trained model artifact
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/shiva-hadi1366/fraud-detection-realtime.git
cd fraud-detection-realtime

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

**1. Train the Model**
```bash
python src/train.py
```

**2. Start the API Server**
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
Access the API docs at: `http://localhost:8000/docs`

**3. Launch the Dashboard**
```bash
streamlit run dashboard/app.py
```
Dashboard available at: `http://localhost:8501`

---

## 🧠 Machine Learning Pipeline

### 1️⃣ Data Preparation
- Load and validate transaction datasets
- Handle missing values & outliers
- Remove duplicates & corrupted records
- Class imbalance handling (SMOTE, class weights)

### 2️⃣ Feature Engineering
Key features created:
- **Transaction Amount** (normalized)
- **Time-based features** (hour, day, month, day of week)
- **Merchant patterns** (transaction frequency, average amount)
- **User behavior** (device changes, location changes)
- **Statistical features** (rolling averages, velocity checks)

### 3️⃣ Model Training
- **Train/Test Split:** 80/20
- **Algorithms:** XGBoost, Random Forest, Gradient Boosting
- **Hyperparameter Tuning:** GridSearch, RandomSearch
- **Cross-Validation:** 5-fold stratified CV

### 4️⃣ Evaluation
- **Precision:** Minimize false positives (customer frustration)
- **Recall:** Minimize false negatives (fraud losses)
- **ROC-AUC:** Overall model discrimination ability
- **Confusion Matrix:** Detailed performance breakdown

---

## 🌐 API Endpoints

### Predict Fraud
```bash
POST /predict
```

**Request Body:**
```json
{
  "transaction_id": "TXN123456",
  "amount": 250.50,
  "merchant_id": "MER789",
  "user_id": "USR456",
  "timestamp": "2024-01-15T14:30:00Z",
  "device_type": "mobile",
  "location": "New York"
}
```

**Response:**
```json
{
  "fraud_probability": 0.87,
  "prediction": 1,
  "risk_level": "high",
  "confidence": 0.95,
  "recommended_action": "block"
}
```

### Health Check
```bash
GET /health
```

### Model Info
```bash
GET /model-info
```

---

## 📊 Dashboard Features

### Pages:
- 📈 **Overview** - Key metrics & fraud statistics
- 🔍 **Transaction Explorer** - Filter & analyze individual transactions
- 📊 **Analytics** - Visualizations & trends
- 🧠 **Real-time Prediction** - Test model with custom inputs
- ⚙️ **Model Performance** - Detailed metrics & confusion matrix

---

## 📈 Example Results

### Model Performance
```
              precision    recall  f1-score   support

         Non-Fraud       0.98      0.96      0.97      4850
              Fraud      0.92      0.95      0.93       150

        accuracy                           0.96      5000
       macro avg       0.95      0.95      0.95      5000
    weighted avg       0.96      0.96      0.96      5000

ROC-AUC Score: 0.9821
```

---

## 🔧 Technologies Used

| Component | Technology |
|-----------|------------|
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **ML Algorithms** | XGBoost, Random Forest, LightGBM |
| **API Framework** | FastAPI, Uvicorn |
| **Dashboard** | Streamlit, Plotly |
| **Database** | (Optional) PostgreSQL, Redis |
| **Containerization** | Docker, Docker Compose |

---

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t fraud-detection:latest .

# Run container
docker run -p 8000:8000 -p 8501:8501 fraud-detection:latest
```

### Cloud Deployment
- **API Server:** AWS Lambda / Google Cloud Functions / Heroku
- **Dashboard:** Streamlit Cloud / AWS Amplify
- **Model Storage:** S3 / GCS / Azure Blob

---

## 📚 Dependencies

See `requirements.txt` for complete list. Key packages:

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
fastapi>=0.95.0
streamlit>=1.20.0
plotly>=5.0.0
pydantic>=1.9.0
```

---

## 🔒 Best Practices

✅ **Security:**
- API authentication (JWT tokens)
- Input validation & sanitization
- Rate limiting on endpoints
- Secure model storage

✅ **Scalability:**
- Async request handling
- Model caching
- Batch prediction support
- Load balancing ready

✅ **Monitoring:**
- Logging & error tracking
- Model performance monitoring
- Alert system for anomalies

---

## 📖 How to Use

1. **For Data Scientists:**
   - Explore notebooks for EDA & experiments
   - Run `src/train.py` to retrain models
   - Adjust hyperparameters in config files

2. **For Backend Developers:**
   - Use FastAPI endpoints for production inference
   - Integrate API into payment systems
   - Monitor via `/health` endpoint

3. **For Business Analysts:**
   - Use Streamlit dashboard for monitoring
   - Generate reports with transaction insights
   - Track fraud trends & patterns

---

## 📈 Future Enhancements

- [ ] Deep Learning models (LSTM, Autoencoders)
- [ ] Real-time feature store integration
- [ ] A/B testing framework
- [ ] Explainability features (SHAP values)
- [ ] Multi-model ensemble approach
- [ ] Kafka/Spark for streaming data
- [ ] Database integration for historical tracking
- [ ] Advanced visualization dashboards

---

## 🐛 Issues & Troubleshooting

**Q: Model takes too long to train?**
- Reduce dataset size for initial testing
- Use `n_jobs=-1` for parallel processing
- Consider using GPU acceleration

**Q: API returns 500 errors?**
- Check model file exists in `models/` directory
- Verify all dependencies are installed
- Check logs for detailed error messages

**Q: Dashboard not loading?**
- Ensure Streamlit is installed: `pip install streamlit`
- Clear browser cache & restart dashboard
- Check console for error messages

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mohammadhadi Shiva**  
🎓 Data Science & Machine Learning Specialist | Python | TensorFlow | PyTorch  
📍 Deutschland  
🔗 [GitHub](https://github.com/shiva-hadi1366) | [LinkedIn](https://linkedin.com/in/shiva-hadi) | [Portfolio](https://shiva-hadi.dev)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support & Contact

- 📧 Email: shiva.hadi1366@gmail.com
- 🐦 Twitter: [@shiva_hadi](https://twitter.com/shiva_hadi)
- 💬 Discord: shiva_hadi#1366
- 📋 Issues: [GitHub Issues](https://github.com/shiva-hadi1366/fraud-detection-realtime/issues)

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Official Guide](https://xgboost.readthedocs.io/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by Mohammadhadi Shiva

</div>