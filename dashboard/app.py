"""
Streamlit Dashboard for Real-Time Fraud Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FraudInference
from src.config import MODEL_PATH, RAW_DATA_PATH

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .fraud-high {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .fraud-medium {
        background-color: #ffaa00;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .fraud-low {
        background-color: #44aa44;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_fraud_model():
    try:
        return FraudInference()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# Load sample data
@st.cache_data
def load_sample_data():
    try:
        if os.path.exists(RAW_DATA_PATH):
            return pd.read_csv(RAW_DATA_PATH)
        else:
            st.warning(f"Data file not found at {RAW_DATA_PATH}")
            return None
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None


# Main app
def main():
    st.title("🔍 Real-Time Fraud Detection System")
    st.markdown("---")

    # Load model
    inference = load_fraud_model()

    if inference is None:
        st.error("❌ Model failed to load. Please check the model path and dependencies.")
        return

    st.success("✅ Model loaded successfully")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Dashboard", "📊 Analytics", "🔮 Make Prediction", "📈 Batch Analysis"]
    )

    # === DASHBOARD PAGE ===
    if page == "🏠 Dashboard":
        dashboard_page(inference)

    # === ANALYTICS PAGE ===
    elif page == "📊 Analytics":
        analytics_page()

    # === PREDICTION PAGE ===
    elif page == "🔮 Make Prediction":
        prediction_page(inference)

    # === BATCH ANALYSIS PAGE ===
    elif page == "📈 Batch Analysis":
        batch_analysis_page(inference)


def dashboard_page(inference):
    """Dashboard overview page"""
    st.header("Dashboard Overview")

    # Load sample data
    df = load_sample_data()

    if df is not None and len(df) > 0:
        col1, col2, col3, col4 = st.columns(4)

        # Metrics
        total_transactions = len(df)
        fraud_count = df["is_fraud"].sum() if "is_fraud" in df.columns else 0
        fraud_rate = (fraud_count / total_transactions * 100) if total_transactions > 0 else 0

        with col1:
            st.metric("Total Transactions", f"{total_transactions:,}")

        with col2:
            st.metric("Fraudulent Cases", f"{fraud_count:,}")

        with col3:
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

        with col4:
            st.metric("Legitimate", f"{total_transactions - fraud_count:,}")

        st.markdown("---")

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            if "is_fraud" in df.columns:
                fraud_dist = df["is_fraud"].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ["#2ecc71", "#e74c3c"]
                ax.bar(["Legitimate", "Fraud"], fraud_dist.values, color=colors)
                ax.set_ylabel("Count")
                ax.set_title("Transaction Distribution")
                st.pyplot(fig)

        with col2:
            if "amount" in df.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(df["amount"], bins=50, color="#3498db", edgecolor="black")
                ax.set_xlabel("Amount")
                ax.set_ylabel("Frequency")
                ax.set_title("Transaction Amount Distribution")
                st.pyplot(fig)

        st.markdown("---")

        # Amount statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Amount", f"€{df['amount'].mean():.2f}")
        with col2:
            st.metric("Max Amount", f"€{df['amount'].max():.2f}")
        with col3:
            st.metric("Min Amount", f"€{df['amount'].min():.2f}")

    else:
        st.warning("No data available. Please run data simulation first.")


def analytics_page():
    """Analytics and detailed analysis page"""
    st.header("Advanced Analytics")

    df = load_sample_data()

    if df is not None and len(df) > 0:
        # Fraud by country
        col1, col2 = st.columns(2)

        with col1:
            if "country" in df.columns and "is_fraud" in df.columns:
                fraud_by_country = df.groupby("country")["is_fraud"].sum().sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                fraud_by_country.plot(kind="barh", ax=ax, color="#e74c3c")
                ax.set_xlabel("Fraud Cases")
                ax.set_title("Fraudulent Transactions by Country")
                st.pyplot(fig)

        with col2:
            if "device" in df.columns and "is_fraud" in df.columns:
                fraud_by_device = df.groupby("device")["is_fraud"].sum()
                fig, ax = plt.subplots(figsize=(10, 6))
                fraud_by_device.plot(kind="bar", ax=ax, color="#3498db")
                ax.set_xlabel("Device Type")
                ax.set_ylabel("Fraud Cases")
                ax.set_title("Fraudulent Transactions by Device")
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # Amount analysis by fraud status
        if "amount" in df.columns and "is_fraud" in df.columns:
            st.subheader("Amount Analysis by Fraud Status")
            fig, ax = plt.subplots(figsize=(12, 6))
            df.boxplot(column="amount", by="is_fraud", ax=ax)
            ax.set_xlabel("Fraud Status (0=Legitimate, 1=Fraud)")
            ax.set_ylabel("Amount")
            ax.set_title("Transaction Amount by Fraud Status")
            st.pyplot(fig)

    else:
        st.warning("No data available.")


def prediction_page(inference):
    """Single transaction prediction page"""
    st.header("🔮 Make Fraud Prediction")

    st.write("Enter transaction details to predict if it's fraudulent:")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.number_input("User ID", min_value=1000, max_value=9999, value=1234)
        device_id = st.number_input("Device ID", min_value=10000, max_value=99999, value=56789)
        amount = st.number_input("Amount (€)", min_value=0.0, value=150.50)
        country = st.selectbox("Country", ["DE", "FR", "IT", "ES", "NL", "PL", "US", "CN"])
        device = st.selectbox("Device Type", ["Mobile", "Web", "ATM", "POS"])

    with col2:
        merchant = st.selectbox("Merchant", ["Electronics", "Clothing", "Food", "Travel", "Crypto", "Gaming"])
        time_since_last = st.number_input("Time Since Last Transaction (seconds)", min_value=0.0, value=3600.0)
        ip_risk_score = st.slider("IP Risk Score", 0.0, 1.0, 0.25, 0.01)
        hour = st.slider("Hour of Day", 0, 23, 14)
        day = st.slider("Day of Month", 1, 31, 15)
        weekday = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)

    if st.button("🔍 Predict", use_container_width=True):
        try:
            transaction_data = {
                "user_id": int(user_id),
                "device_id": int(device_id),
                "amount": float(amount),
                "country": country,
                "device": device,
                "merchant": merchant,
                "time_since_last": float(time_since_last),
                "ip_risk_score": float(ip_risk_score),
                "hour": int(hour),
                "day": int(day),
                "weekday": int(weekday)
            }

            result = inference.predict(transaction_data)

            # Display result
            fraud_prob = result["fraud_probability"]
            fraud_label = result["fraud_label"]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Fraud Probability", f"{fraud_prob:.1%}")

            with col2:
                if fraud_label == 1:
                    st.error(f"⚠️ FRAUDULENT")
                else:
                    st.success(f"✅ LEGITIMATE")

            # Risk level indicator
            st.markdown("---")
            if fraud_prob < 0.3:
                st.markdown('<div class="fraud-low"><b>🟢 Low Risk</b></div>', unsafe_allow_html=True)
            elif fraud_prob < 0.7:
                st.markdown('<div class="fraud-medium"><b>🟡 Medium Risk</b></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="fraud-high"><b>🔴 High Risk</b></div>', unsafe_allow_html=True)

            # Details
            st.subheader("Transaction Details")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Amount:** €{amount:.2f}")
                st.write(f"**Country:** {country}")
                st.write(f"**Device:** {device}")
            with col2:
                st.write(f"**Merchant:** {merchant}")
                st.write(f"**Time Since Last:** {time_since_last/3600:.1f} hours")
                st.write(f"**IP Risk Score:** {ip_risk_score:.2f}")

        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")


def batch_analysis_page(inference):
    """Batch analysis page"""
    st.header("📈 Batch Transaction Analysis")

    df = load_sample_data()

    if df is not None and len(df) > 0:
        st.write(f"Total transactions in dataset: **{len(df):,}**")

        # Get sample
        sample_size = st.slider("Sample Size for Analysis", 10, min(100, len(df)), 50)

        if st.button("Analyze Sample", use_container_width=True):
            sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

            st.write(f"Analyzing {len(sample_df)} transactions...")

            progress_bar = st.progress(0)
            results = []

            for idx, row in sample_df.iterrows():
                try:
                    transaction = {
                        "user_id": int(row["user_id"]),
                        "device_id": int(row["device_id"]),
                        "amount": float(row["amount"]),
                        "country": row["country"],
                        "device": row["device"],
                        "merchant": row["merchant"],
                        "time_since_last": float(row["time_since_last"]),
                        "ip_risk_score": float(row["ip_risk_score"]),
                        "hour": int(row["hour"]),
                        "day": int(row["day"]),
                        "weekday": int(row["weekday"])
                    }

                    prediction = inference.predict(transaction)
                    results.append({
                        "Amount": row["amount"],
                        "Country": row["country"],
                        "Device": row["device"],
                        "Fraud Probability": prediction["fraud_probability"],
                        "Predicted Label": "Fraud" if prediction["fraud_label"] == 1 else "Legitimate",
                        "Actual Label": "Fraud" if row["is_fraud"] == 1 else "Legitimate"
                    })
                except:
                    pass

                progress_bar.progress((idx + 1) / len(sample_df))

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)

            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predictions Made", len(results_df))
            with col2:
                fraud_predicted = (results_df["Predicted Label"] == "Fraud").sum()
                st.metric("Fraud Predicted", fraud_predicted)
            with col3:
                avg_prob = results_df["Fraud Probability"].mean()
                st.metric("Avg Fraud Probability", f"{avg_prob:.1%}")

    else:
        st.warning("No data available. Please run data simulation first.")


if __name__ == "__main__":
    main()
