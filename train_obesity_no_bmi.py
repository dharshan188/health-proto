# train_obesity_no_bmi.py
"""
Train obesity model WITHOUT BMI using available Pima numeric features.
This is ONLY for prototype/demo because dataset lacks lifestyle features.
"""

import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_FP = 'data/merged.csv'
df = pd.read_csv(DATA_FP)
print("Loaded merged.csv ->", df.shape)

# Only keep rows with obesity label (PIMA dataset)
d = df[df['obesity'].notna()].copy()

# Features WITHOUT BMI
features = ['age', 'systolic_bp', 'fasting_glucose']
print("Using features:", features)

X = d[features]
y = d['obesity'].astype(int)

stratify = y if len(set(y)) > 1 else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# Preprocessor
preproc = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), features)
])

clf = Pipeline([
    ('preproc', preproc),
    ('rf', RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight='balanced'
    ))
])

print("\nTraining obesity-no-BMI model...")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

if len(set(y_test)) > 1:
    y_prob = clf.predict_proba(X_test)[:, 1]
    print("ROC AUC:", roc_auc_score(y_test, y_prob))

joblib.dump(clf, "obesity_no_bmi_pipeline.joblib")
print("\nSaved model: obesity_no_bmi_pipeline.joblib")
