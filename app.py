"""A small web demo.  Run:  streamlit run app.py     (this file is what you deploy on Hugging Face Spaces)"""
import joblib
import pandas as pd
import streamlit as st

from features import BILL_COLS, PAY_AMT_COLS, PAY_COLS, add_features
from logger import log_prediction

st.title("Credit Default Risk Demo")
st.write("Enter a client's details. The model estimates the chance of missing next month's payment.")

model = joblib.load("models/model.joblib")

limit = st.number_input("Credit limit", 10000, 1000000, 100000, step=10000)
age = st.slider("Age", 21, 79, 35)
pay_status = st.slider("Repayment status (-1 = paid on time, 1+ = months late)", -1, 8, 0)
bill = st.number_input("Typical monthly bill", 0, 1000000, 50000, step=5000)
paid = st.number_input("Typical monthly payment", 0, 1000000, 5000, step=1000)
sex = st.selectbox("Sex (1 = male, 2 = female)", [1, 2])
education = st.selectbox("Education (1 = graduate, 2 = university, 3 = high school, 4 = other)", [1, 2, 3, 4], index=1)
marriage = st.selectbox("Marriage (1 = married, 2 = single, 3 = other)", [1, 2, 3])

if st.button("Predict risk"):
    # For simplicity the demo uses the same value for each of the 6 months
    row = {"LIMIT_BAL": limit, "SEX": sex, "EDUCATION": education, "MARRIAGE": marriage, "AGE": age}
    row.update({col: pay_status for col in PAY_COLS})
    row.update({col: bill for col in BILL_COLS})
    row.update({col: paid for col in PAY_AMT_COLS})
    row = add_features(pd.DataFrame([row]))
    probability = float(model.predict_proba(row)[0, 1])
    log_prediction(row, probability)
    st.metric("Estimated default risk", f"{probability:.1%}")
    st.write("High risk - review this client." if probability >= 0.5 else "Lower risk.")
