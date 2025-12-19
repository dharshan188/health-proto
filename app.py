import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and related objects
obj = joblib.load('multi_disease_model.joblib')
pipelines = obj['pipelines']
feature_sets = obj['feature_sets']
targets = obj['targets']

# Get the union of all feature names
all_feats = sorted(list(set(f for fset in feature_sets for f in fset)))

# Define categorical feature options
categorical_options = {
    'sex': ['M', 'F'],
    'fasting_blood_sugar': [0, 1],
    'exercise_induced_angina': [0, 1],
    'rest_ecg': ['normal', 'st-t abnormality', 'lv hypertrophy'],
    'slope': ['upsloping', 'flat', 'downsloping'],
    'thalassemia': ['normal', 'fixed defect', 'reversable defect'],
    'chest_pain_type': ['typical', 'atypical', 'non-anginal', 'asymptomatic'],
    'vessels_colored_by_flourosopy': [0, 1, 2, 3, 4]
}

def main():
    st.set_page_config(page_title="Health-Proto", layout="wide")
    st.title("❤️‍🩹 Health-Proto: Multi-Disease Risk Prediction")
    st.write("This tool predicts your risk for Diabetes, Obesity, and Heart Disease using a machine learning model. Please enter your health information below.")

    st.sidebar.header("Enter Your Health Data")

    sample = {}
    for feature in all_feats:
        if feature in categorical_options:
            sample[feature] = st.sidebar.selectbox(f"**{feature.replace('_', ' ').title()}**", categorical_options[feature])
        else:
            # Determine a reasonable range and default for numeric inputs
            min_val, max_val, default_val, step = 0.0, 300.0, 25.0, 1.0
            if feature == 'age':
                min_val, max_val, default_val = 1.0, 120.0, 50.0
            elif feature == 'BMI':
                min_val, max_val, default_val = 10.0, 60.0, 25.0
            elif feature == 'systolic_bp':
                min_val, max_val, default_val = 80.0, 200.0, 120.0
            elif feature == 'fasting_glucose':
                min_val, max_val, default_val = 50.0, 300.0, 100.0
            elif feature == 'resting_blood_pressure':
                min_val, max_val, default_val = 80.0, 220.0, 120.0
            elif feature == 'cholestoral':
                min_val, max_val, default_val = 100.0, 600.0, 200.0
            elif feature == 'Max_heart_rate':
                min_val, max_val, default_val = 60.0, 220.0, 150.0
            elif feature == 'oldpeak':
                min_val, max_val, default_val, step = 0.0, 10.0, 1.0, 0.1

            sample[feature] = st.sidebar.number_input(f"**{feature.replace('_', ' ').title()}**", min_value=min_val, max_value=max_val, value=default_val, step=step)

    if st.button("Analyze My Health Risk", key="predict_button"):
        with st.spinner('Analyzing...'):
            # Create a DataFrame from the user input
            X = pd.DataFrame([sample], columns=all_feats)

            # Make predictions
            preds = {}
            probs = {}
            for i, target in enumerate(targets):
                feats = feature_sets[i]
                pipe = pipelines[i]
                Xsub = X[feats]
                preds[target] = int(pipe.predict(Xsub)[0])
                try:
                    probs[target] = float(pipe.predict_proba(Xsub)[:, 1][0])
                except Exception:
                    probs[target] = None

            st.success("Analysis Complete!")

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

if __name__ == "__main__":
    main()
