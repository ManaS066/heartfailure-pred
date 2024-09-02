import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Set page configuration
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("Heart Failure Prediction System")
st.markdown("""
This application predicts the likelihood of death based on clinical records using a **Decision Tree Regression** model.
""")

# Load the dataset (hidden from the user)
@st.cache_data
def load_data():
    dataset = pd.read_csv('heart_failure_clinical_records.csv')
    dataset = dataset.drop(columns=['age'])
    return dataset

dataset = load_data()

# Feature engineering and model training (hidden from the user)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

# Evaluate the model using R² score (hidden from the user)
y_pred = regressor.predict(X_test)
r2 = r2_score(y_test, y_pred)

# User input for prediction
st.write("### Patient Outcome Prediction")
st.write("Enter the patient's clinical data below:")

with st.form(key='prediction_form'):
    # Dropdown for sex selection
    sex = st.selectbox("Select Gender", options=["Male", "Female"])
    sex_value = 0 if sex == "Male" else 1
    
    # Dropdown for diabetes
    diabetes = st.selectbox("Has Diabetes?", options=["Yes", "No"])
    diabetes_value = 1 if diabetes == "Yes" else 0

    # Dropdown for smoking status
    smoking = st.selectbox("Smoking Status", options=["Non-Smoker", "Smoker"])
    smoking_value = 1 if smoking == "Smoker" else 0

    # Other feature inputs
    feature_inputs = {}
    for feature in dataset.columns[:-1]:
        if feature not in ['sex', 'diabetes', 'smoking']:  # Exclude handled features
            feature_inputs[feature] = st.number_input(f"Enter value for {feature}", value=0, step=1)  # Whole number input

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Collect input values, including sex, diabetes, and smoking status
        input_values = np.array([[sex_value, diabetes_value, smoking_value] + list(feature_inputs.values())])
        input_values_scaled = sc_X.transform(input_values)
        prediction = regressor.predict(input_values_scaled)
        
        outcome = 'Death' if prediction[0] == 1 else 'Survival'
        
        st.success(f"The patient is likely to experience **{outcome}**.")
