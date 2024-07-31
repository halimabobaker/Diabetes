import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('xgb_model_with_smote.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize LabelEncoders if needed
diet_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
family_history_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()

# Fit the encoders (assuming you have the training data)
diet_encoder.fit(['Always', 'Sometimes'])
bp_encoder.fit(['Yes', 'No'])
family_history_encoder.fit(['Yes', 'No'])
smoking_encoder.fit(['Non-smoker', 'Smoker'])

# Function to make predictions
def predict_diabetes(Age, BMI, Diet, BP, Family_History, Smoking):
    # Encode categorical inputs
    try:
        encoded_diet = diet_encoder.transform([Diet])[0]
        encoded_bp = bp_encoder.transform([BP])[0]
        encoded_family_history = family_history_encoder.transform([Family_History])[0]
        encoded_smoking = smoking_encoder.transform([Smoking])[0]
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        return None, None
    
    # Create input data array
    input_data = np.array([[Age, BMI, encoded_diet, encoded_bp, encoded_family_history, encoded_smoking]])
    
    # Make prediction
    try:
        prediction_proba = model.predict_proba(input_data)[0][1]  # Get probability of positive class
        return prediction_proba, 'High Risk of Diabetes' if prediction_proba > 0.5 else 'Low Risk of Diabetes'
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# Streamlit app
st.title("Diabetes Prediction in Libya")
st.sidebar.header("User Input Parameters")
Age = st.sidebar.number_input('Choose Age', min_value=0, max_value=100)
BMI = st.sidebar.number_input('Choose Body Mass Index (BMI)', min_value=0, max_value=50)
Diet = st.sidebar.selectbox('How regular do you eat fruits and vegetables?', options=['Always', 'Sometimes'])
BP = st.sidebar.selectbox('Have you ever taken hypertension medicine?', options=['Yes', 'No'])
Family_History = st.sidebar.selectbox('Have you ever had a Family History?', options=['Yes', 'No'])
Smoking = st.sidebar.selectbox('Smoking habits?', options=['Non-smoker', 'Smoker'])

if st.sidebar.button('Predict'):
    prediction_proba, risk = predict_diabetes(Age, BMI, Diet, BP, Family_History, Smoking)
    if prediction_proba is not None:
        st.write(f"The predicted probability of High Risk of Type 2 Diabetes Mellitus is: {prediction_proba:.2f}")
        st.write(f"Risk Assessment: {risk}")
        
        # Remarks based on prediction probability
        if prediction_proba > 0.5:
            st.write("""
                **Remarks:**
                - If the predicted Score label is “Yes” and the scored probability is greater than 0.5 and closer to 1.0, then we advise the person to consult a good diabetologist to avoid further progression of the disease.
                - If the probability score is equal to 0.5, the person should make necessary dietary precautions (consult a dietician) and try to avoid a sedentary lifestyle (join a gym and burn the extra fats and carbohydrates stored in your body).
                - If the scoring probability is close to zero, a big thumbs up for the person! Keep up the good work and encourage others to stay healthy and wise like yourself.
            """)
        else:
            st.write("""
                **Remarks:**
                - Your current score indicates a low risk of diabetes, but it is still important to maintain a healthy lifestyle and regular check-ups to ensure long-term health.
            """)
