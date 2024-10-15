import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Gradient Boosting model
filename_gb = 'gradient_boost_model.sav'
model_gb = pickle.load(open(filename_gb, 'rb'))

# Define the input fields
def get_user_input():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.selectbox('SeniorCitizen', ['Yes', 'No'])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    phone_service = st.selectbox('PhoneService', ['Yes', 'No'])
    online = st.selectbox('Online', ['Yes', 'No'])
    security = st.selectbox('Security', ['Yes', 'No'])
    online_backup = st.selectbox('OnlineBackup', ['Yes', 'No'])
    device_protection = st.selectbox('DeviceProtection', ['Yes', 'No'])
    tech_support = st.selectbox('TechSupport', ['Yes', 'No'])
    streaming_tv = st.selectbox('StreamingTV', ['Yes', 'No'])
    streaming_movies = st.selectbox('StreamingMovies', ['Yes', 'No'])
    paperless_billing = st.selectbox('PaperlessBilling', ['Yes', 'No'])
    tenure = st.slider('Tenure', 1, 72)
    internet = st.selectbox('InternetService', ['DSL', 'Fiber optic', 'None'])
    service = st.selectbox('ServiceType', ['Month-to-month', 'One year', 'Two year'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox('PaymentMethod', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthly_charge = st.number_input('MonthlyCharge', min_value=0.0, format="%.2f")
    total_charge = st.number_input('TotalCharge', min_value=0.0, format="%.2f")

    # Convert categorical features to numeric
    data = {
        'gender': 1 if gender == 'Female' else 0,
        'senior_citizen': 1 if senior_citizen == 'Yes' else 0,
        'partner': 1 if partner == 'Yes' else 0,
        'dependents': 1 if dependents == 'Yes' else 0,
        'phone_service': 1 if phone_service == 'Yes' else 0,
        'online': 1 if online == 'Yes' else 0,
        'security': 1 if security == 'Yes' else 0,
        'online_backup': 1 if online_backup == 'Yes' else 0,
        'device_protection': 1 if device_protection == 'Yes' else 0,
        'tech_support': 1 if tech_support == 'Yes' else 0,
        'streaming_tv': 1 if streaming_tv == 'Yes' else 0,
        'streaming_movies': 1 if streaming_movies == 'Yes' else 0,
        'paperless_billing': 1 if paperless_billing == 'Yes' else 0,
        'tenure': tenure,
        'internet': internet,
        'service': service,
        'contract': contract,
        'payment_method': payment_method,
        'monthly_charge': monthly_charge,
        'total_charge': total_charge
    }
    
    # Convert categorical features to one-hot encoded
    df = pd.DataFrame(data, index=[0])
    
    return df

# Create Streamlit interface
st.title('Customer Churn Prediction')

user_input = get_user_input()

# Predict
if st.button('Predict'):
    prediction = model_gb.predict(user_input)
    probability = model_gb.predict_proba(user_input)
    st.write(f"Churn Probability: {probability[0][1]:.2f}")
    st.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")
