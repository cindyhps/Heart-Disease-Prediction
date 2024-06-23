import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PowerTransformer
from scipy.stats import boxcox

# Load the trained models
with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

# Load the accuracy models
with open('model_accuracies.pkl', 'rb') as f:
    model_accuracies = pickle.load(f)

# Define a function for prediction
def predict(model, data):
    return model.predict_proba(data)

# Define a function to transform input data
def transform_input(data, lambdas):
    pt = PowerTransformer()
    for feature in lambdas.keys():
        data[feature] = pt.fit_transform(data[[feature]].apply(lambda x: boxcox(x, lambdas[feature])))
    return data

# Set up the Streamlit interface
st.title('Heart Disease Prediction')

st.sidebar.header('User Input Features')

# Define the input fields
def user_input_features():
    age = st.sidebar.slider('Age', 10, 90, 50)
    sex = st.sidebar.selectbox('Sex, Male 0 and Female 1', [0, 1])
    cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (mg/dL)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dL', [0, 1])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
    ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', [0, 1, 2, 3])

    # Filter hanya fitur yang relevan
    data = {
        'age': age,
        'sex': sex,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal_1': 0,  
        'thal_2': 0,
        'thal_3': 0,
        'cp_1': 0,
        'cp_2': 0,
        'cp_3': 0,
        'restecg_1': 0,
        'restecg_2': 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the input fields and corresponding descriptions
def main():
    st.title('Heart Disease Prediction App')
    st.sidebar.header('User Input Features')

    # Display input fields
    input_df = user_input_features()
    st.sidebar.subheader('Input features:')
    st.sidebar.write(input_df)

    # Display variable descriptions
    st.subheader('Variable Descriptions:')
    st.write("""
    - **age**: Age of the patient in years
    - **sex**: Gender of the patient (0 = male, 1 = female)
    - **cp**: Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)
    - **trestbps**: Resting blood pressure in mm Hg
    - **chol**: Serum cholesterol in mg/dl
    - **fbs**: Fasting blood sugar level (1 = true, 0 = false)
    - **restecg**: Resting electrocardiographic results (0: Normal, 1: ST-T Wave Abnormality, 2: Probable or Definite Left Ventricular Hypertrophy)
    - **thalach**: Maximum heart rate achieved during a stress test
    - **exang**: Exercise induced angina (1 = yes, 0 = no)
    - **oldpeak**: ST depression induced by exercise relative to rest
    - **slope**: Slope of the peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)
    - **ca**: Number of major vessels colored by fluoroscopy (0-4)
    - **thal**: Thalium stress test result (0: Normal, 1: Fixed Defect, 2: Reversible Defect, 3: Not Described)
    """)


# Transform input data
lambdas = {'age': 0.5, 'trestbps': 0.1, 'chol': 0.1, 'thalach': 0.1, 'oldpeak': 0.1}  # Example lambdas, replace with actual values
input_transformed = transform_input(input_df.copy(), lambdas)

# Scaling input data
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_transformed)

# Sidebar options for model selection
model_choice = st.sidebar.selectbox('Model Choice', ['DT', 'RF', 'KNN', 'SVM'])

# Prediction
if st.button('Predict'):
    model = models[model_choice]
    accuracy = model_accuracies[model_choice]
    prediction = predict(model, input_scaled)
    st.write(f'Prediction probability of having heart disease: {prediction[0][1]*100:.2f}%')
    st.write(f'Confidence level (based on model accuracy): {accuracy*100:.2f}%')

# Display user input
st.subheader('User Input Features')
st.write(input_df)
