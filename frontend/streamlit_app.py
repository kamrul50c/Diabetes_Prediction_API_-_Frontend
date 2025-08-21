import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "https://diabetes-prediction-api-frontend-oiyj.onrender.com") 

st.set_page_config(page_title="Diabetes Prediction")
st.title("Diabetes Prediction")

with st.form("patient_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3, step=1)
        Glucose = st.number_input("Glucose", min_value=0.0, value=145.0)
        BloodPressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0)
    with col2:
        SkinThickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0)
        Insulin = st.number_input("Insulin", min_value=0.0, value=85.0)
        BMI = st.number_input("BMI", min_value=0.0, value=33.6)
    with col3:
        DiabetesPedigreeFunction = st.number_input("DPF", min_value=0.0, value=0.35, step=0.01, format="%.2f")
        Age = st.number_input("Age", min_value=0, max_value=120, value=29, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Pregnancies": Pregnancies, "Glucose": Glucose, "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness, "Insulin": Insulin, "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction, "Age": Age
    }
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=20)
        if r.ok:
            res = r.json()
            st.success(f"Prediction: **{res['result']}** (class={res['prediction']}, confidence={res['confidence']})")
        else:
            st.error(f"API error {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")

with st.expander("Model metrics"):
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=10)
        if r.ok:
            st.json(r.json())
        else:
            st.write("Metrics not available yet.")
    except Exception:
        st.write("Metrics not available.")
