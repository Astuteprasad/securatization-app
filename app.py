import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Securitisation Investment Decision Tool")

st.write("Adjust the inputs below:")

# Inputs
RP = st.slider("Profitability (RP)", 0.0, 1.0, 0.5)
SR = st.slider("Stress Resilience (SR)", 0.0, 1.0, 0.5)
SS = st.slider("Stability (SS)", 0.0, 1.0, 0.5)
L = st.slider("Liquidity (L)", 0.0, 1.0, 0.5)
CQ = st.slider("Credit Quality (CQ)", 0.0, 1.0, 0.5)

if st.button("Evaluate"):

    new_data = pd.DataFrame([[RP, SR, SS, L, CQ]],
                            columns=['RP','SR','SS','L','CQ'])

    new_data_scaled = scaler.transform(new_data)

    prediction = model.predict(new_data_scaled)[0]
    probability = model.predict_proba(new_data_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ INVEST (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ REJECT (Confidence: {1-probability:.2f})")
