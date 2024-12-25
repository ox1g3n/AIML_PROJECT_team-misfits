import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
import os
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.0001, gamma=0.95, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.001):
        self.model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

st.set_page_config(
    page_title="Medical Diagnosis System",
    page_icon="🏥",
    layout="wide"
)

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the disease",
    ["Heart Disease", "COVID-19"]
)

if app_mode == "Heart Disease":
    
    st.title("❤ Heart Disease Severity Prediction")
    st.write("""
    This application helps predict the severity of heart disease based on various medical indicators.
    Please enter the values for different symptoms to get a prediction.
    """)

    @st.cache_resource
    def load_model():
        data = pd.read_csv('merged_heart_disease_data.csv')
        X = data.drop(columns=['num'])
        scaler = StandardScaler()
        scaler.fit(X)
        with open('dqn_healthcare_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, scaler, X.columns.tolist()

    try:
        model, scaler, feature_names = load_model()
        
        col1, col2 = st.columns(2)
        
        user_inputs = {}
        
        for i, feature in enumerate(feature_names):
            if i < len(feature_names) // 2:
                user_inputs[feature] = col1.number_input(
                    f"{feature}",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.1
                )
            else:
                user_inputs[feature] = col2.number_input(
                    f"{feature}",
                    min_value=0.0,
                    max_value=1000.0,
                    value=0.0,
                    step=0.1
                )
        
        if st.button("Predict Severity"):
            input_values = [user_inputs[feature] for feature in feature_names]
            scaled_inputs = scaler.transform([input_values])[0]
            prediction = model.choose_action(scaled_inputs)
            
            severity_descriptions = {
                0: "No heart disease",
                1: "Mild severity",
                2: "Moderate severity",
                3: "High severity",
                4: "Critical severity"
            }
            
            # Display prediction
            st.subheader("Prediction Result")
            
            # Create colored box based on severity
            if prediction == 0:
                st.success(f"Severity Level: {prediction} - {severity_descriptions[prediction]}")
            elif prediction == 1:
                st.info(f"Severity Level: {prediction} - {severity_descriptions[prediction]}")
            elif prediction == 2:
                st.warning(f"Severity Level: {prediction} - {severity_descriptions[prediction]}")
            elif prediction == 3:
                st.error(f"Severity Level: {prediction} - {severity_descriptions[prediction]}")
            else:  # prediction == 4
                st.error(f"⚠ Severity Level: {prediction} - {severity_descriptions[prediction]}")
            
            # Display recommendations based on severity
            st.subheader("Recommendations")
            
            if prediction == 0:
                st.write("Recommendations for Maintaining Heart Health:")
                st.write("• Continue your healthy lifestyle habits")
                st.write("• Regular exercise (30 minutes of moderate activity, 5 days a week)")
                st.write("• Maintain a balanced diet rich in fruits and vegetables")
                st.write("• Regular health check-ups once a year")
                st.write("• Maintain healthy sleep habits (7-9 hours per night)")
                st.write("• Manage stress through relaxation techniques")
                
            elif prediction == 1:
                st.write("Recommendations for Mild Risk:")
                st.write("• Schedule a check-up with your healthcare provider within the next month")
                st.write("• Monitor your blood pressure regularly")
                st.write("• Reduce salt intake to less than 5g per day")
                st.write("• Start a moderate exercise program (after consulting your doctor)")
                st.write("• Consider lifestyle modifications to reduce risk factors")
                st.write("• Keep a health diary to track symptoms")
                
            elif prediction == 2:
                st.warning("⚠ Recommendations for Moderate Risk:")
                st.write("• Schedule an appointment with a cardiologist within 2 weeks")
                st.write("• Begin daily blood pressure monitoring")
                st.write("• Strict adherence to a heart-healthy diet")
                st.write("• Reduce work-related stress if possible")
                st.write("• Consider cardiac rehabilitation programs")
                st.write("• Have emergency contact numbers readily available")
                st.write("• Discuss medication options with your healthcare provider")
                
            elif prediction == 3:
                st.error("🚨 Urgent Recommendations for High Risk:")
                st.write("• Seek medical attention within the next 24-48 hours")
                st.write("• Contact your healthcare provider immediately")
                st.write("• Monitor symptoms closely and keep a detailed log")
                st.write("• Have someone stay with you for support")
                st.write("• Prepare a list of current medications for your doctor")
                st.write("• Avoid strenuous physical activity until cleared by doctor")
                st.write("• Keep emergency numbers readily accessible")
                st.write("• Consider wearing a medical alert device")
                
            else:  # prediction == 4
                st.error("🚨 IMMEDIATE ACTION REQUIRED - Critical Risk:")
                st.write("• SEEK EMERGENCY MEDICAL CARE IMMEDIATELY")
                st.write("• Call emergency services (911) if experiencing:")
                st.write("  - Chest pain or pressure")
                st.write("  - Difficulty breathing")
                st.write("  - Severe fatigue")
                st.write("  - Irregular heartbeat")
                st.write("• Do not drive yourself to the hospital")
                st.write("• Take aspirin if recommended by medical professionals")
                st.write("• Keep a list of current medications ready for emergency responders")
                st.write("• Inform family members or emergency contacts")
            
            # Add disclaimer
            st.caption("""
                Disclaimer: This prediction is based on a machine learning model and should not be used as the sole basis
                for medical decisions. Always consult with a healthcare professional for proper medical advice.
            """)

    except Exception as e:
        st.error(f"""
            Error loading the model: {str(e)}
            
            Please make sure the following files exist in the same directory as this script:
            - merged_heart_disease_data.csv
            - dqn_healthcare_model.pkl
        """)

    # Add information about the project
    with st.expander("About this Project"):
        st.write("""
            This project uses a Deep Q-Network (DQN) to predict heart disease severity based on various medical indicators.
            The model was trained on a dataset containing multiple heart disease indicators and their corresponding severity levels.
            
            The severity levels range from 0 to 4:
            - 0: No heart disease
            - 1: Mild severity
            - 2: Moderate severity
            - 3: High severity
            - 4: Critical severity
            
            Please note that this is a predictive model and should not replace professional medical advice.
        """)
    
elif app_mode == "COVID-19":
    st.title("COVID-19 Classification System")
    st.write("Enter patient information to predict COVID-19 classification")

    try:
        data = pd.read_csv('covid_55k.csv')
        data = data.drop(columns=['DATE_DIED'])
        X = data.drop(columns=['CLASIFFICATION_FINAL'])
        
        scaler = StandardScaler()
        scaler.fit(X)
        
        with open('dqn_classification2_model.pkl', 'rb') as f:
            covid_model = pickle.load(f)
        
        features = X.columns.tolist()
        
        col1, col2 = st.columns(2)
        user_inputs = {}
        
        for i, feature in enumerate(features):
            if i < len(features) // 2:
                user_inputs[feature] = col1.number_input(
                    feature,
                    value=0,
                    step=1,
                    help=f"Enter value for {feature}"
                )
            else:
                user_inputs[feature] = col2.number_input(
                    feature,
                    value=0,
                    step=1,
                    help=f"Enter value for {feature}"
                )

        if st.button("Predict"):
            input_values = [user_inputs[feature] for feature in features]
            scaled_inputs = scaler.transform([input_values])[0]
            prediction = covid_model.choose_action(scaled_inputs)
            
            classifications = {
                1: "Positive COVID-19",
                2: "Negative COVID-19",
                3: "Suspected COVID-19"
            }
            
            result = classifications.get(prediction + 1, "Unknown")
            
            if result == "Positive COVID-19":
                st.error(f"Classification Result: {result}")
            elif result == "Negative COVID-19":
                st.success(f"Classification Result: {result}")
            else:
                st.warning(f"Classification Result: {result}")
            
            st.warning("This is a preliminary classification. Please consult healthcare professionals for proper diagnosis.")
    except Exception as e:
        st.error(f"Error: {str(e)}\nPlease ensure Covid Data 55k.csv and dqn_classification2_model.pkl are present.")