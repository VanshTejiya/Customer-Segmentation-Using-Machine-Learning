import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load model & scaler
# ----------------------------
try:
    kmeans = joblib.load(r"C:\Users\vansh\Desktop\python\Customer Segmentaion\kmeans_model.pkl")
    scaler = joblib.load(r"C:\Users\vansh\Desktop\python\Customer Segmentaion\scaler.pkl")
except:
    st.error("❌ Failed to load model or scaler. Check file paths!")
    st.stop()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🎯 Customer Segmentation App")
st.write("Fill the details below to predict which customer segment this person belongs to.")

with st.expander("🧍 Customer Information", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income", min_value=1000, max_value=300000, value=50000)
        recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)

    with col2:
        total_spending = st.number_input("Total Spending (last 2 years)", min_value=0, max_value=10000, value=1200)
        num_web_purchases = st.number_input("Web Purchases", min_value=0, max_value=200, value=10)
        num_store_purchases = st.number_input("Store Purchases", min_value=0, max_value=200, value=8)
        num_web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=100, value=5)

# ----------------------------
# Prepare input
# ----------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})

# ----------------------------
# Prediction button
# ----------------------------
if st.button("🔍 Predict Segment"):
    with st.spinner("Analyzing customer and predicting segment..."):
        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        cluster = kmeans.predict(input_scaled)[0]

    # Show result
    st.success(f"🎉 Predicted Segment: **Cluster {cluster}**")

    # ----------------------------
    # Optional: Cluster meaning
    # ----------------------------
    cluster_info = {
        0: "🟦 **High Income, High Spending, Engaged Customers**",
        1: "🟩 **Low Income, Low Spending, Less Engaged**",
        2: "🟧 **Young, Medium Income, Medium Spending**",
        3: "🟨 **High Web Engagement Users**",
        4: "🟥 **Customers with High Recency (inactive recently)**"
    }

    if cluster in cluster_info:
        st.info(cluster_info[cluster])

    # Optional debugging
    with st.expander("🔧 Debug Info (Scaled Input)"):
        st.write(pd.DataFrame(input_scaled, columns=input_data.columns))



#streamlit run segementation.py