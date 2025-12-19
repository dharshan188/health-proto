import streamlit as st
import joblib
import pandas as pd
import os

# Load the model and related objects
obj = joblib.load('multi_disease_model.joblib')
pipelines = obj['pipelines']
feature_sets = obj['feature_sets']
targets = obj['targets']
all_feats = sorted(list(set(f for fset in feature_sets for f in fset)))
PATIENTS_FILE = 'data/patients.csv'

def doctor_dashboard():
    st.title("Doctor Dashboard")

    if os.path.exists(PATIENTS_FILE):
        df_patients = pd.read_csv(PATIENTS_FILE)
        total_patients = len(df_patients)
        high_risk_count = 0
        for target in targets:
            if target in df_patients.columns:
                high_risk_count += df_patients[target].sum()
    else:
        total_patients = 0
        high_risk_count = 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="High Risk", value=high_risk_count)

    with col2:
        st.metric(label="Total Patients", value=total_patients)


    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Add Patient", key="add_patient_dashboard"):
        st.session_state.page = "add_patient"
        st.rerun()
    if st.button("View Families", key="view_families"):
        st.session_state.page = "view_families"
        st.rerun()
    if st.button("View Risk Result", key="view_risk_result_dashboard"):
        st.info("Please add a patient first to view their risk result.")

def add_patient():
    st.title("Add Patient")

    # Mappings from UI friendly names to model values
    chest_pain_map = {
        'Typical Angina': 'typical',
        'Atypical Angina': 'atypical',
        'Non-Anginal Pain': 'non-anginal',
        'Asymptomatic': 'asymptomatic'
    }
    rest_ecg_map = {
        'Normal': 'normal',
        'ST-T Wave Abnormality': 'st-t abnormality',
        'Left Ventricular Hypertrophy': 'lv hypertrophy'
    }

    with st.form(key="patient_form"):
        st.header("Patient Details")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name")
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", ["Male", "Female"])
        with col2:
            blood_group = st.text_input("Blood Group")
            contact_number = st.text_input("Contact Number")

        st.header("Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=10, max_value=200, value=70)
        with col2:
            systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
            cholestoral = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
        with col3:
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)", min_value=50, max_value=400, value=100)
            Max_heart_rate = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)

        st.header("Clinical Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            chest_pain_type_ui = st.selectbox("Chest Pain Type", list(chest_pain_map.keys()))
            rest_ecg_ui = st.selectbox("Resting ECG", list(rest_ecg_map.keys()))
            exercise_induced_angina = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col2:
            oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope", ['upsloping', 'flat', 'downsloping'])
            thalassemia = st.selectbox("Thalassemia", ['normal', 'fixed defect', 'reversable defect'])
        with col3:
             vessels_colored_by_flourosopy = st.selectbox("Vessels Colored by Flourosopy", [0, 1, 2, 3, 4])

        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        st.session_state.patient_name = full_name
        patient_data = {}
        patient_data['age'] = age
        patient_data['sex'] = 'M' if sex == 'Male' else 'F'
        if height > 0:
            patient_data['BMI'] = weight / ((height / 100) ** 2)
        else:
            patient_data['BMI'] = 25
        patient_data['systolic_bp'] = systolic_bp
        patient_data['resting_blood_pressure'] = systolic_bp
        patient_data['cholestoral'] = cholestoral
        patient_data['fasting_glucose'] = fasting_glucose
        patient_data['fasting_blood_sugar'] = 1 if fasting_glucose > 120 else 0
        patient_data['Max_heart_rate'] = Max_heart_rate
        patient_data['chest_pain_type'] = chest_pain_map[chest_pain_type_ui]
        patient_data['rest_ecg'] = rest_ecg_map[rest_ecg_ui]
        patient_data['exercise_induced_angina'] = exercise_induced_angina
        patient_data['oldpeak'] = oldpeak
        patient_data['slope'] = slope
        patient_data['thalassemia'] = thalassemia
        patient_data['vessels_colored_by_flourosopy'] = vessels_colored_by_flourosopy

        X = pd.DataFrame([patient_data], columns=all_feats)

        preds = {}
        probs = {}
        for i, target in enumerate(targets):
            feats = feature_sets[i]
            pipe = pipelines[i]
            Xsub = X[feats].copy()
            preds[target] = int(pipe.predict(Xsub)[0])
            try:
                probs[target] = float(pipe.predict_proba(Xsub)[:, 1][0])
            except Exception:
                probs[target] = None

        # Save patient data
        patient_record = {'full_name': full_name, **patient_data, **preds}
        df_new = pd.DataFrame([patient_record])

        if not os.path.exists(PATIENTS_FILE):
            df_new.to_csv(PATIENTS_FILE, index=False)
        else:
            df_new.to_csv(PATIENTS_FILE, mode='a', header=False, index=False)

        st.session_state.predictions = (preds, probs)
        st.session_state.page = "view_risk_result"
        st.rerun()


def view_risk_result():
    patient_name = st.session_state.get("patient_name", "Patient")
    st.title(f"Risk Result for {patient_name}")

    if 'predictions' in st.session_state:
        preds, probs = st.session_state.predictions

        st.subheader("Prediction Results")
        cols = st.columns(len(targets))
        for idx, target in enumerate(targets):
            with cols[idx]:
                st.metric(
                    label=f"**{target.replace('_', ' ').title()}**",
                    value="High Risk" if preds[target] == 1 else "Low Risk",
                    delta=f"{probs[target]:.0%}" if probs[target] is not None else "N/A",
                    delta_color="inverse"
                )
        st.info("The percentages indicate the model's confidence in the 'High Risk' prediction.")
    else:
        st.error("There was an error in generating the prediction. Please check the input data.")

    if st.button("Back to Dashboard"):
        st.session_state.page = "doctor_dashboard"
        if 'predictions' in st.session_state:
            del st.session_state.predictions
        if 'patient_name' in st.session_state:
            del st.session_state.patient_name
        st.rerun()

def view_families():
    st.title("View Families")
    st.info("This feature is currently under development.")
    if st.button("Back to Dashboard"):
        st.session_state.page = "doctor_dashboard"
        st.rerun()


def main():
    st.set_page_config(page_title="Family Disease Risk App", layout="wide")

    if 'page' not in st.session_state:
        st.session_state.page = "doctor_dashboard"

    if st.session_state.page == "doctor_dashboard":
        doctor_dashboard()
    elif st.session_state.page == "add_patient":
        add_patient()
    elif st.session_state.page == "view_risk_result":
        view_risk_result()
    elif st.session_state.page == "view_families":
        view_families()

if __name__ == "__main__":
    main()
