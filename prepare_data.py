# prepare_data.py
import pandas as pd, numpy as np
# Load Pima (diabetes)
pima = pd.read_csv('data/pima.csv')   # ensure path and filename
# Typical Pima columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
# Normalize names:
pima = pima.rename(columns={
    'Glucose':'fasting_glucose',
    'BMI':'BMI',
    'Age':'age',
    'BloodPressure':'systolic_bp' # NOTE: Pima's BloodPressure is actually diastolic; treat as generic BP if needed
})
pima['sex'] = np.nan
pima['source'] = 'pima'
# Create diabetes label from Outcome if exists or derive using hba1c/fg
if 'Outcome' in pima.columns:
    pima['diabetes'] = pima['Outcome'].astype(int)
else:
    pima['diabetes'] = (pima['fasting_glucose'] >= 126).astype(int)

# obesity label from BMI if present
if 'BMI' in pima.columns:
    pima['obesity'] = (pima['BMI'] >= 30).astype(int)
else:
    pima['obesity'] = np.nan

# small cleanup: keep only needed columns
cols_needed = ['age','sex','BMI','systolic_bp','fasting_glucose','diabetes','obesity','source']
pima = pima[[c for c in cols_needed if c in pima.columns]]

# Load heart dataset (UCI) - ensure it's in data/heart.csv
heart = pd.read_csv('data/heart.csv')
# Common columns may be named: age, sex, trestbps(systolic), chol(total_chol), target
heart = heart.rename(columns={
    'trestbps':'systolic_bp',
    'chol':'total_chol',
    'target':'heart_disease'
})
heart['source'] = 'heart'
# Create heart_disease binary (if target exists)
if 'heart_disease' in heart.columns:
    heart['heart_disease'] = (heart['heart_disease'] > 0).astype(int)

# Ensure BMI exists in heart; if not, fill with NaN
if 'BMI' not in heart.columns:
    heart['BMI'] = np.nan
# diabetes label may not exist; leave NaN if not present
if 'diabetes' not in heart.columns:
    heart['diabetes'] = np.nan
if 'obesity' not in heart.columns:
    heart['obesity'] = (heart['BMI'] >= 30).astype(int) if heart['BMI'].notna().any() else np.nan

# Load obesity dataset
obesity_df = pd.read_csv('data/obesity.csv')
obesity_df = obesity_df.rename(columns={
    'Gender':'sex',
    'Age':'age',
    'family_history_with_overweight':'family_history_overweight'
})
# Create target: 1 if not normal weight
obesity_df['obesity'] = (obesity_df['NObeyesdad'] != 'Normal_Weight').astype(int)
obesity_df['source'] = 'obesity'
# Normalize sex values
obesity_df['sex'] = obesity_df['sex'].str[0].str.upper()

# Standardize columns
common_cols = list(set(pima.columns) | set(heart.columns) | set(obesity_df.columns))
merged = pd.concat([pima, heart, obesity_df], ignore_index=True, sort=False)
# Save
merged.to_csv('data/merged.csv', index=False)
print("Merged saved to data/merged.csv -- rows:", len(merged))
