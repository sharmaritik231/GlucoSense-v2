import streamlit as st
import pandas as pd
import commons  # Make sure to import your module that contains generate_data, perform_feature_selection, perform_diabetes_test, and perform_bgl_test
import seaborn as sns
import matplotlib.pyplot as plt

# Increase the width of the content for all pages
st.set_page_config(layout="wide")

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.selectbox("Go to", ["Home", "Report"])

    if selection == "Home":
        show_home()
    elif selection == "Report":
        show_report()

def show_home():
    st.title("GlucoSense: A non-invasive Blood Glucose monitor")
    st.markdown("""
    GlucoSense is an AI-powered web application designed to predict diabetes non-invasively using breath-based sensor data and physiological parameters. This tool leverages machine learning algorithms to classify individuals into three categories—non-diabetic, prediabetic, and highly diabetic—based on volatile organic compound (VOC) responses from breath samples, along with body vitals and demographic details.

    ## Key Features:
    - **Non-Invasive Prediction** – Eliminates the need for painful blood tests by analyzing breath samples.
    - **Advanced Machine Learning** – Utilizes frequency-domain filtering (FFT) and feature extraction for high-accuracy classification.
    - **Real-Time Data Processing** – Connects with an ESP32 microcontroller to wirelessly receive and analyze data.
    - **User-Friendly Interface** – Provides a seamless experience for healthcare professionals and researchers.
    """)
    
    # Creating columns
    col1, col2 = st.columns(2)

    # Input fields with default values in columns
    with col1:
        age = st.number_input("Age", min_value=0, value=30)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
    with col2:
        spo2 = st.number_input("SPO2", min_value=0, max_value=100, value=95)
        heart_rate = st.number_input("Heart Rate", min_value=0, value=70)
        
    # Convert gender to numeric and store personal information in session state
    gender_numeric = 0 if gender == "Male" else 1
    
    # Store personal information in session state
    st.session_state["name"] = "User"  # Default name since not captured in form
    st.session_state["age"] = age
    st.session_state["gender"] = gender_numeric
    st.session_state["heart_rate"] = heart_rate
    st.session_state["spo2"] = spo2
    
    # Initialize default BP values
    max_bp, min_bp = 120, 80  # Default normal BP values
    
    # File upload
    uploaded_file = st.file_uploader("Breath Response", type=["csv"])

    if uploaded_file is not None:
        # Create initial body vitals DataFrame for data generation
        body_vitals = {'Age': [age], 'Gender': [gender_numeric], 'maxBP': [max_bp], 'minBP': [min_bp], 'HR': [heart_rate], 'SPO2': [spo2]}
        body_vitals = pd.DataFrame(body_vitals)
        
        # Read the CSV file
        data = pd.read_csv(uploaded_file, skiprows=3).iloc[:, 1:]

        # Generate data for the test
        test_data = commons.generate_data(data, body_vitals)
        max_bp, min_bp = commons.perform_bp_test(test_data)
        diabetes_result, bgl_result = commons.perform_diabetes_test(test_data)

        # Store results in session state to access in the report page
        st.session_state["diabetes_result"] = diabetes_result
        st.session_state["bgl_result"] = bgl_result
        st.session_state["max_bp"] = max_bp
        st.session_state["min_bp"] = min_bp
        st.success("Test Completed! Go to the 'Report' page to see the results.")

    else:
        st.warning("Please upload a CSV file.")
        # Store default BP values when no file is uploaded
        st.session_state["max_bp"] = max_bp
        st.session_state["min_bp"] = min_bp
        
    # Create final body vitals DataFrame with predicted/default BP values
    body_vitals = {'Age': [age], 'Gender': [gender_numeric], 'maxBP': [max_bp], 'minBP': [min_bp], 'HR': [heart_rate], 'SPO2': [spo2]}
    body_vitals = pd.DataFrame(body_vitals)
    st.session_state["body_vitals"] = body_vitals

def show_report():
    st.title("GlucoSense: Health Report")
    st.markdown("""
    The Diabetes Report page in GlucoSense provides a detailed analysis of an individual's health status based on breath-based sensor data and physiological parameters. 
    This report offers valuable insights into diabetes classification—non-diabetic, prediabetic, or highly diabetic—using advanced machine learning techniques.
    """)

    if "diabetes_result" in st.session_state and "bgl_result" in st.session_state:
        # Personal Information Section
        st.subheader("Personal Information and Body Vitals", divider=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Age", value=f"{st.session_state['age']}", border=True)
        with col2:
            st.metric(label="Gender", value='Male' if st.session_state['gender'] == 0 else 'Female', border=True)
        with col3:
            st.metric(label="Heart Rate", value=f"{st.session_state['heart_rate']} bpm", border=True)
        with col4:
            st.metric(label="SPO2 (%)", value=f"{st.session_state['spo2']}", border=True)

        # Diabetes Prediction Section
        st.subheader("Health Status", divider=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Systolic BP", value=f"{st.session_state['max_bp']} mmHg", border=True)
        with col2:
            st.metric(label="Diastolic BP", value=f"{st.session_state['min_bp']} mmHg", border=True)
        with col3:
            st.metric(label="BGL Severity", value=st.session_state['diabetes_result'], border=True)
        with col4:
            st.metric(label="Blood Glucose Level (mg/dL)", value=f"{st.session_state['bgl_result']}", border=True)

    else:
        st.warning("Please fill your details on Home Page.")

if __name__ == "__main__":
    main()
