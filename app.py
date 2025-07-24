# Streamlit app for Vendor Fraud Detection using saved .pkl models for scaler and Random Forest

import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Vendor Fraud Detection App")

uploaded_file = st.file_uploader("Upload Amazon.csv", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['InvoiceHour'] = df['InvoiceDate'].dt.hour
    
    features = ['Amount', 'InvoiceHour']
    X = df[features].fillna(0)

    # Load pre-trained scaler and Random Forest model
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('fraud_rf_model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Scale data and predict using Random Forest
    X_scaled = scaler.transform(X)
    df['RF_Pred'] = clf.predict(X_scaled)
    
    # Run Isolation Forest for anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['AnomalyFlag'] = iso.fit_predict(X_scaled)
    df['AnomalyFlag'] = df['AnomalyFlag'].apply(lambda x: 1 if x == -1 else 0)

    # Combine results for potential fraud
    df['PotentialFraud'] = df.apply(lambda x: 1 if x['AnomalyFlag'] == 1 or x['RF_Pred'] == 1 else 0, axis=1)

    st.success(f"Analysis complete. Total flagged fraud transactions: {df['PotentialFraud'].sum()}")
    st.write("## Flagged Fraud Transactions")
    st.dataframe(df[df['PotentialFraud'] == 1])

    st.write("## Fraud Detection Scatter Plot")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.scatterplot(data=df, x='Amount', y='InvoiceHour', hue='PotentialFraud', palette='coolwarm', ax=ax)
    plt.title('Vendor Fraud Detection - Flagged Transactions')
    st.pyplot(fig)

    csv = df[df['PotentialFraud'] == 1].to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Flagged Fraud Transactions CSV",
        data=csv,
        file_name='flagged_fraud_transactions.csv',
        mime='text/csv'
    )

