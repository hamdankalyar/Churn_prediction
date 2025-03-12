# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from preprocessing import load_preprocessing_objects, preprocess_input  # Import from preprocessing.py

# Load the trained ANN model
model = tf.keras.models.load_model('./models/ann_model.h5')

# Load preprocessing objects using the function from preprocessing.py
one_hot_encoder_geo, label_encoder_gender, scaler = load_preprocessing_objects()

# Streamlit UI (example, you'll need to add your actual UI elements)
st.title("Churn Prediction")

# Example input fields (replace with your actual UI)
credit_score = st.number_input("Credit Score", value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", value=40)
tenure = st.number_input("Tenure", value=5)
balance = st.number_input("Balance", value=0)
num_of_products = st.number_input("Number of Products", value=2)
has_cr_card = st.selectbox("Has Credit Card", [1, 0])
is_active_member = st.selectbox("Is Active Member", [1, 0])
estimated_salary = st.number_input("Estimated Salary", value=80000)

# Prepare input data
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Preprocess the input data using the function from preprocessing.py
input_scaled = preprocess_input(input_data, one_hot_encoder_geo, label_encoder_gender, scaler)

# Make prediction (example)
prediction = model.predict(input_scaled)
st.write("Prediction:", prediction) # display the prediction.

# Make prediction
prediction_probability = model.predict(input_scaled)[0][0] # Get prediction probability
# Explanation:
# - `model.predict(input_scaled)`: Makes the prediction using the loaded model.
# - `[0][0]`: Extracts the probability value from the output array.

churn_threshold = 0.5 # Set churn probability threshold
# Explanation:
# - `churn_threshold = 0.5`: Defines the threshold for classifying churn (0.5 is a common default for binary classification).

# Display Prediction and Probability
st.write("### Prediction:") # Heading for the prediction section
st.write(f"Churn Probability: {prediction_probability:.4f}") # Display churn probability with 4 decimal places
# Explanation:
# - `st.write("### Prediction:")`: Displays a level-3 heading in the Streamlit app.
# - `st.write(f"Churn Probability: {prediction_probability:.4f}")`: Displays the churn probability, formatted to 4 decimal places.

if prediction_probability > churn_threshold:
    st.write("### Customer is likely to churn.") # Display churn prediction if probability is above threshold
else:
    st.write("### Customer is not likely to churn.") # Display no-churn prediction otherwise
# Explanation:
# - `if prediction_probability > churn_threshold:`: Checks if the predicted probability is greater than the threshold.
#   - `st.write("### Customer is likely to churn.")`: If true, displays "Customer is likely to churn."
#   - `else: st.write("### Customer is not likely to churn.")`: Otherwise, displays "Customer is not likely to churn."