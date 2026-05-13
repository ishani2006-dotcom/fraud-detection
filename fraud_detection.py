import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")
st.title("fraud detection prediction app")
st.markdown("please enter the transaction detailes and use the predict buttons")
st.divider()
transaction_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER","CASH_OUT", "DEPOSIT"])
amount = st.number_input("amount", min_value= 0.0, value= 1000.0)
oldbalanceOrg = st.number_input("old balance(sender)", min_value= 0.0, value= 1000.0)
oldbalanceDest = st.number_input("old balance(receiver)", min_value= 0.0, value = 0.0)
newbalanceDest = st.number_input("new balance(receiver)", min_value=0.0 , value= 0.0)
if st.button("Predict"):
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount" : amount,
        "oldbalanceOrg": oldbalanceOrg,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]
    st.subheader(f"Prediction : '{int(prediction)}'")

    if prediction == 1:
        st.error("this transaction can be fraud")
    else:
        st.success("this Transaction looks like it is not a fraud")
    