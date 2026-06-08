import streamlit as st
import pandas as pd
import joblib

model=joblib.load("../Models/churn_model.pkl")

st.title("Customer Churn Prediction")

gender=st.selectbox("Gender",["Male","Female"])
seniorcitizen=st.selectbox("Senior Citizen",[0,1])
partner=st.selectbox("Partner",["Yes","No"])
dependents=st.selectbox("Dependents",["Yes","No"])
tenure=st.slider("Tenure",0,72)

phoneservice=st.selectbox("Phone Service",["Yes","No"])
multiplelines=st.selectbox("Multiple Lines",["Yes","No","No phone service"])
internetservice=st.selectbox("Internet Service",["DSL","Fiber optic","No"])

onlinesecurity=st.selectbox("Online Security",["Yes","No","No internet service"])
onlinebackup=st.selectbox("Online Backup",["Yes","No","No internet service"])
deviceprotection=st.selectbox("Device Protection",["Yes","No","No internet service"])

techsupport=st.selectbox("Tech Support",["Yes","No","No internet service"])
streamingtv=st.selectbox("Streaming TV",["Yes","No","No internet service"])
streamingmovies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])

contract=st.selectbox("Contract",["Month-to-month","One year","Two year"])

paperlessbilling=st.selectbox("Paperless Billing",["Yes","No"])

paymentmethod=st.selectbox(
    "Payment Method",["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)

monthlycharges=st.number_input("Monthly Charges")
totalcharges=st.number_input("Total Charges")

if st.button("Predict"):

    input_df=pd.DataFrame({

        "Gender":[gender],
        "Seniorcitizen":[seniorcitizen],
        "Partner":[partner],
        "Dependents":[dependents],
        "Tenure":[tenure],
        "Phoneservice":[phoneservice],
        "Multiplelines":[multiplelines],
        "Internetservice":[internetservice],
        "Onlinesecurity":[onlinesecurity],
        "Onlinebackup":[onlinebackup],
        "Deviceprotection":[deviceprotection],
        "Techsupport":[techsupport],
        "Streamingtv":[streamingtv],
        "Streamingmovies":[streamingmovies],
        "Contract":[contract],
        "Paperlessbilling":[paperlessbilling],
        "Paymentmethod":[paymentmethod],
        "Monthlycharges":[monthlycharges],
        "Totalcharges":[totalcharges]

    })

    probability=model.predict_proba(input_df)[0][1]

    prediction=int(probability>0.2)

    if prediction==1:

        st.error("Customer likely to churn")

        st.write(f"Churn Probability: {probability:.2%}")

    else:

        st.success("Customer likely to stay")

        st.write(f"Stay Probability: {(1-probability):.2%}")