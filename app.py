import streamlit as st
import joblib
import pandas as pd
import os
import numpy as np
from datetime import datetime

# Load the model and related objects
obj = joblib.load('multi_disease_model.joblib')
pipelines = obj['pipelines']
feature_sets = obj['feature_sets']
targets = obj['targets']
all_feats = sorted(list(set(f for fset in feature_sets for f in fset)))

PATIENTS_FILE = 'data/patients.csv'
FAMILIES_FILE = 'data/families.csv'

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

def calculate_genetic_risk_modifier(family_history):
    """Calculate genetic risk modifier based on family history"""
    modifier = 1.0
    if family_history.get('parents_affected', 0) > 0:
        modifier += 0.3 * family_history['parents_affected']
    if family_history.get('siblings_affected', 0) > 0:
        modifier += 0.2 * family_history['siblings_affected']
    if family_history.get('grandparents_affected', 0) > 0:
        modifier += 0.1 * family_history['grandparents_affected']
    return min(modifier, 2.0)  # Cap at 2x risk

def predict_multi_horizon_risk(base_prob, age, genetic_modifier):
    """Predict risk over 1, 5, and 10 year horizons"""
    # Age progression factor
    age_factor = 1 + (0.01 * (age - 40))  # Risk increases with age
    
    # 1-year risk (current)
    risk_1yr = min(base_prob * genetic_modifier, 0.99)
    
    # 5-year risk (compound probability with progression)
    risk_5yr = min(1 - (1 - risk_1yr) ** (5 * age_factor), 0.99)
    
    # 10-year risk (compound probability with progression)
    risk_10yr = min(1 - (1 - risk_1yr) ** (10 * age_factor * 1.1), 0.99)
    
    return risk_1yr, risk_5yr, risk_10yr

def doctor_dashboard():
    st.title("🏥 Doctor Dashboard - Disease Risk Assessment System")
    
    # Load patient data
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
        df_patients = None
    
    # Load family data
    if os.path.exists(FAMILIES_FILE):
        df_families = pd.read_csv(FAMILIES_FILE)
        total_families = len(df_families)
    else:
        total_families = 0
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="👥 Total Patients", value=total_patients)
    with col2:
        st.metric(label="⚠️ High Risk Cases", value=high_risk_count)
    with col3:
        st.metric(label="👨‍👩‍👧‍👦 Families", value=total_families)
    with col4:
        avg_risk = (high_risk_count / (total_patients * len(targets)) * 100) if total_patients > 0 else 0
        st.metric(label="📊 Avg Risk %", value=f"{avg_risk:.1f}%")
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("➕ Add Patient", use_container_width=True):
            st.session_state.page = "add_patient"
            st.rerun()
    with col2:
        if st.button("👀 View Patients", use_container_width=True):
            st.session_state.page = "view_patients"
            st.rerun()
    with col3:
        if st.button("👨‍👩‍👧 Manage Families", use_container_width=True):
            st.session_state.page = "view_families"
            st.rerun()
    with col4:
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
    
    st.markdown("---")
    
    # Recent patients
    if df_patients is not None and len(df_patients) > 0:
        st.subheader("📋 Recent Patients")
        recent = df_patients.tail(5)[['full_name', 'age', 'sex'] + [t for t in targets if t in df_patients.columns]]
        st.dataframe(recent, use_container_width=True, hide_index=True)

def add_patient():
    st.title("➕ Add New Patient")
    
    # Load families for dropdown
    families = []
    if os.path.exists(FAMILIES_FILE):
        df_families = pd.read_csv(FAMILIES_FILE)
        families = df_families['family_name'].tolist()
    
    # Mappings
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
        st.header("👤 Patient Details")
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name*", placeholder="Enter patient's full name")
            age = st.number_input("Age*", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex*", ["Male", "Female"])
        with col2:
            blood_group = st.text_input("Blood Group", placeholder="e.g., O+, A-, B+")
            contact_number = st.text_input("Contact Number", placeholder="Enter phone number")
            family_name = st.selectbox("Family", ["None"] + families)
        
        st.header("📏 Vitals")
        col1, col2, col3 = st.columns(3)
        with col1:
            height = st.number_input("Height (cm)*", min_value=50, max_value=250, value=170)
            weight = st.number_input("Weight (kg)*", min_value=10, max_value=200, value=70)
        with col2:
            systolic_bp = st.number_input("Systolic BP*", min_value=80, max_value=200, value=120)
            cholestoral = st.number_input("Cholesterol (mg/dL)*", min_value=100, max_value=400, value=200)
        with col3:
            fasting_glucose = st.number_input("Fasting Glucose (mg/dL)*", min_value=50, max_value=400, value=100)
            Max_heart_rate = st.number_input("Max Heart Rate*", min_value=60, max_value=220, value=150)
        
        st.header("🔬 Clinical Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            chest_pain_type_ui = st.selectbox("Chest Pain Type*", list(chest_pain_map.keys()))
            rest_ecg_ui = st.selectbox("Resting ECG*", list(rest_ecg_map.keys()))
            exercise_induced_angina = st.selectbox("Exercise Induced Angina*", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        with col2:
            oldpeak = st.number_input("Oldpeak*", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
            slope = st.selectbox("Slope*", ['upsloping', 'flat', 'downsloping'])
            thalassemia = st.selectbox("Thalassemia*", ['normal', 'fixed defect', 'reversable defect'])
        with col3:
            vessels_colored_by_flourosopy = st.selectbox("Vessels Colored by Flourosopy*", [0, 1, 2, 3, 4])
        
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button(label="✅ Submit & Analyze", use_container_width=True)
        with col2:
            cancel_button = st.form_submit_button(label="❌ Cancel", use_container_width=True)
    
    if cancel_button:
        st.session_state.page = "doctor_dashboard"
        st.rerun()
    
    if submit_button:
        if not full_name.strip():
            st.error("Please enter patient's full name")
            return
        
        st.session_state.patient_name = full_name
        patient_data = {}
        patient_data['age'] = age
        patient_data['sex'] = 'M' if sex == 'Male' else 'F'
        patient_data['BMI'] = weight / ((height / 100) ** 2) if height > 0 else 25
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
                probs[target] = 0.5
        
        # Get family history if family is selected
        genetic_modifier = 1.0
        if family_name != "None" and os.path.exists(FAMILIES_FILE):
            df_fam = pd.read_csv(FAMILIES_FILE)
            fam_row = df_fam[df_fam['family_name'] == family_name]
            if not fam_row.empty:
                family_history = {
                    'parents_affected': fam_row.iloc[0].get('parents_affected', 0),
                    'siblings_affected': fam_row.iloc[0].get('siblings_affected', 0),
                    'grandparents_affected': fam_row.iloc[0].get('grandparents_affected', 0)
                }
                genetic_modifier = calculate_genetic_risk_modifier(family_history)
        
        # Save patient data
        patient_record = {
            'full_name': full_name,
            'blood_group': blood_group,
            'contact_number': contact_number,
            'family_name': family_name if family_name != "None" else "",
            'date_added': datetime.now().strftime("%Y-%m-%d"),
            **patient_data,
            **preds
        }
        df_new = pd.DataFrame([patient_record])
        
        if not os.path.exists(PATIENTS_FILE):
            df_new.to_csv(PATIENTS_FILE, index=False)
        else:
            df_new.to_csv(PATIENTS_FILE, mode='a', header=False, index=False)
        
        st.session_state.predictions = (preds, probs, genetic_modifier, age)
        st.session_state.page = "view_risk_result"
        st.rerun()

def view_risk_result():
    patient_name = st.session_state.get("patient_name", "Patient")
    st.title(f"📊 Risk Assessment for {patient_name}")
    
    if 'predictions' in st.session_state:
        preds, probs, genetic_modifier, age = st.session_state.predictions
        
        st.info(f"🧬 Genetic Risk Modifier: {genetic_modifier:.2f}x (Based on family history)")
        
        st.subheader("🎯 Current Risk Assessment")
        cols = st.columns(len(targets))
        for idx, target in enumerate(targets):
            with cols[idx]:
                risk_status = "🔴 High Risk" if preds[target] == 1 else "🟢 Low Risk"
                st.metric(
                    label=f"{target.replace('_', ' ').title()}",
                    value=risk_status,
                    delta=f"{probs[target]:.0%}" if probs[target] is not None else "N/A"
                )
        
        st.markdown("---")
        st.subheader("📅 Multi-Horizon Risk Prediction")
        
        for target in targets:
            if probs[target] is not None:
                base_prob = probs[target]
                risk_1yr, risk_5yr, risk_10yr = predict_multi_horizon_risk(base_prob, age, genetic_modifier)
                
                st.markdown(f"**{target.replace('_', ' ').title()}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("1-Year Risk", f"{risk_1yr*100:.1f}%")
                with col2:
                    st.metric("5-Year Risk", f"{risk_5yr*100:.1f}%")
                with col3:
                    st.metric("10-Year Risk", f"{risk_10yr*100:.1f}%")
                st.progress(risk_10yr)
                st.markdown("")
        
        st.markdown("---")
        st.success("✅ Patient data saved successfully!")
    else:
        st.error("No prediction data available. Please add a patient first.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.session_state.page = "doctor_dashboard"
            if 'predictions' in st.session_state:
                del st.session_state.predictions
            if 'patient_name' in st.session_state:
                del st.session_state.patient_name
            st.rerun()
    with col2:
        if st.button("➕ Add Another Patient", use_container_width=True):
            st.session_state.page = "add_patient"
            if 'predictions' in st.session_state:
                del st.session_state.predictions
            if 'patient_name' in st.session_state:
                del st.session_state.patient_name
            st.rerun()

def view_patients():
    st.title("👥 Patient List")
    
    if not os.path.exists(PATIENTS_FILE):
        st.warning("No patients found. Add your first patient to get started.")
        if st.button("➕ Add Patient"):
            st.session_state.page = "add_patient"
            st.rerun()
        return
    
    df_patients = pd.read_csv(PATIENTS_FILE)
    
    if len(df_patients) == 0:
        st.warning("No patients found.")
        return
    
    # Search and filter
    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search by name", placeholder="Enter patient name...")
    with col2:
        risk_filter = st.selectbox("Filter by Risk", ["All", "High Risk", "Low Risk"])
    
    # Apply filters
    filtered_df = df_patients.copy()
    if search:
        filtered_df = filtered_df[filtered_df['full_name'].str.contains(search, case=False, na=False)]
    
    if risk_filter == "High Risk":
        mask = filtered_df[targets].sum(axis=1) > 0
        filtered_df = filtered_df[mask]
    elif risk_filter == "Low Risk":
        mask = filtered_df[targets].sum(axis=1) == 0
        filtered_df = filtered_df[mask]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df_patients)} patients**")
    
    # Display patients
    display_cols = ['full_name', 'age', 'sex', 'blood_group', 'contact_number', 'family_name', 'date_added']
    display_cols = [col for col in display_cols if col in filtered_df.columns]
    
    for idx, row in filtered_df.iterrows():
        with st.expander(f"👤 {row['full_name']} | Age: {row['age']} | Sex: {row['sex']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Personal Information**")
                st.write(f"Blood Group: {row.get('blood_group', 'N/A')}")
                st.write(f"Contact: {row.get('contact_number', 'N/A')}")
                st.write(f"Family: {row.get('family_name', 'None')}")
                st.write(f"BMI: {row.get('BMI', 'N/A'):.1f}" if 'BMI' in row else "BMI: N/A")
            
            with col2:
                st.write("**Vitals**")
                st.write(f"BP: {row.get('systolic_bp', 'N/A')}")
                st.write(f"Cholesterol: {row.get('cholestoral', 'N/A')}")
                st.write(f"Glucose: {row.get('fasting_glucose', 'N/A')}")
                st.write(f"Heart Rate: {row.get('Max_heart_rate', 'N/A')}")
            
            st.write("**Risk Assessment**")
            risk_cols = st.columns(len(targets))
            for i, target in enumerate(targets):
                if target in row:
                    with risk_cols[i]:
                        risk_val = row[target]
                        st.metric(
                            target.replace('_', ' ').title(),
                            "High" if risk_val == 1 else "Low",
                            delta_color="inverse"
                        )
    
    if st.button("🏠 Back to Dashboard"):
        st.session_state.page = "doctor_dashboard"
        st.rerun()

def view_families():
    st.title("👨‍👩‍👧‍👦 Family Management")
    
    tab1, tab2 = st.tabs(["📋 View Families", "➕ Add Family"])
    
    with tab1:
        if not os.path.exists(FAMILIES_FILE):
            st.info("No families registered yet. Add a family to track genetic risk factors.")
        else:
            df_families = pd.read_csv(FAMILIES_FILE)
            if len(df_families) == 0:
                st.info("No families found.")
            else:
                for idx, row in df_families.iterrows():
                    with st.expander(f"👨‍👩‍👧 {row['family_name']} ({row.get('num_members', 0)} members)"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Family History**")
                            st.write(f"Parents Affected: {row.get('parents_affected', 0)}")
                            st.write(f"Siblings Affected: {row.get('siblings_affected', 0)}")
                            st.write(f"Grandparents Affected: {row.get('grandparents_affected', 0)}")
                        with col2:
                            st.write("**Notes**")
                            st.write(row.get('notes', 'No notes'))
                        
                        # Show family members
                        if os.path.exists(PATIENTS_FILE):
                            df_patients = pd.read_csv(PATIENTS_FILE)
                            family_members = df_patients[df_patients['family_name'] == row['family_name']]
                            if len(family_members) > 0:
                                st.write(f"**Family Members ({len(family_members)})**")
                                st.dataframe(
                                    family_members[['full_name', 'age', 'sex']],
                                    hide_index=True,
                                    use_container_width=True
                                )
    
    with tab2:
        with st.form("add_family_form"):
            st.subheader("Add New Family")
            family_name = st.text_input("Family Name*", placeholder="e.g., Smith Family")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                parents_affected = st.number_input("Parents with Disease History", min_value=0, max_value=2, value=0)
            with col2:
                siblings_affected = st.number_input("Siblings with Disease History", min_value=0, max_value=10, value=0)
            with col3:
                grandparents_affected = st.number_input("Grandparents with Disease History", min_value=0, max_value=4, value=0)
            
            notes = st.text_area("Notes", placeholder="Any additional family medical history...")
            
            submit = st.form_submit_button("✅ Add Family", use_container_width=True)
            
            if submit:
                if not family_name.strip():
                    st.error("Please enter a family name")
                else:
                    family_record = {
                        'family_name': family_name,
                        'parents_affected': parents_affected,
                        'siblings_affected': siblings_affected,
                        'grandparents_affected': grandparents_affected,
                        'notes': notes,
                        'num_members': 0,
                        'date_added': datetime.now().strftime("%Y-%m-%d")
                    }
                    df_new = pd.DataFrame([family_record])
                    
                    if not os.path.exists(FAMILIES_FILE):
                        df_new.to_csv(FAMILIES_FILE, index=False)
                    else:
                        df_new.to_csv(FAMILIES_FILE, mode='a', header=False, index=False)
                    
                    st.success(f"✅ Family '{family_name}' added successfully!")
                    st.rerun()
    
    if st.button("🏠 Back to Dashboard"):
        st.session_state.page = "doctor_dashboard"
        st.rerun()

def analytics():
    st.title("📈 Analytics Dashboard")
    
    if not os.path.exists(PATIENTS_FILE):
        st.warning("No patient data available for analysis.")
        if st.button("🏠 Back to Dashboard"):
            st.session_state.page = "doctor_dashboard"
            st.rerun()
        return
    
    df_patients = pd.read_csv(PATIENTS_FILE)
    
    if len(df_patients) == 0:
        st.warning("No patients found.")
        return
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_age = df_patients['age'].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    with col2:
        avg_bmi = df_patients['BMI'].mean() if 'BMI' in df_patients.columns else 0
        st.metric("Average BMI", f"{avg_bmi:.1f}")
    with col3:
        male_count = (df_patients['sex'] == 'M').sum()
        st.metric("Male/Female Ratio", f"{male_count}/{len(df_patients)-male_count}")
    
    st.markdown("---")
    
    # Disease prevalence
    st.subheader("🎯 Disease Risk Distribution")
    disease_stats = []
    for target in targets:
        if target in df_patients.columns:
            high_risk = df_patients[target].sum()
            total = len(df_patients)
            percentage = (high_risk / total * 100) if total > 0 else 0
            disease_stats.append({
                'Disease': target.replace('_', ' ').title(),
                'High Risk Count': high_risk,
                'Percentage': f"{percentage:.1f}%"
            })
    
    if disease_stats:
        st.dataframe(pd.DataFrame(disease_stats), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Age distribution
    st.subheader("👥 Age Distribution")
    age_bins = [0, 30, 40, 50, 60, 70, 120]
    age_labels = ['<30', '30-40', '40-50', '50-60', '60-70', '70+']
    df_patients['age_group'] = pd.cut(df_patients['age'], bins=age_bins, labels=age_labels)
    age_dist = df_patients['age_group'].value_counts().sort_index()
    st.bar_chart(age_dist)
    
    if st.button("🏠 Back to Dashboard"):
        st.session_state.page = "doctor_dashboard"
        st.rerun()

def main():
    st.set_page_config(
        page_title="Family Disease Risk Prediction",
        page_icon="🏥",
        layout="wide"
    )
    
    # Add custom CSS
    st.markdown("""
        <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if 'page' not in st.session_state:
        st.session_state.page = "doctor_dashboard"
    
    # Sidebar
    with st.sidebar:
        st.title("🏥 Disease Risk App")
        st.markdown("---")
        st.markdown("**Navigation**")
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = "doctor_dashboard"
            st.rerun()
        if st.button("➕ Add Patient", use_container_width=True):
            st.session_state.page = "add_patient"
            st.rerun()
        if st.button("👥 View Patients", use_container_width=True):
            st.session_state.page = "view_patients"
            st.rerun()
        if st.button("👨‍👩‍👧 Families", use_container_width=True):
            st.session_state.page = "view_families"
            st.rerun()
        if st.button("📈 Analytics", use_container_width=True):
            st.session_state.page = "analytics"
            st.rerun()
        
        st.markdown("---")
        st.markdown("**About**")
        st.info("This system predicts disease risk using ML models combined with family genetic history for 1, 5, and 10-year horizons.")
    
    # Route to pages
    if st.session_state.page == "doctor_dashboard":
        doctor_dashboard()
    elif st.session_state.page == "add_patient":
        add_patient()
    elif st.session_state.page == "view_risk_result":
        view_risk_result()
    elif st.session_state.page == "view_patients":
        view_patients()
    elif st.session_state.page == "view_families":
        view_families()
    elif st.session_state.page == "analytics":
        analytics()

if __name__ == "__main__":
    main()
