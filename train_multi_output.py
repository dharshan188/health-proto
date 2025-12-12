# train_multi_output.py
"""
Stable per-target pipelines (no global ColumnTransformer mismatch):
 - Diabetes  (uses BMI)
 - Obesity   (NO BMI)
 - Heart     (uses BMI)

Saves: multi_disease_model.joblib containing:
{'pipelines': [pipe_diabetes, pipe_obesity, pipe_heart],
 'feature_sets': feature_sets,
 'targets': targets}
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

DATA_FP = "data/merged.csv"
if not os.path.exists(DATA_FP):
    raise SystemExit("data/merged.csv not found.")

df = pd.read_csv(DATA_FP)
print("Loaded:", df.shape)

# Ensure targets exist and fill missing with 0 (assume negative)
targets = ['diabetes', 'obesity', 'heart_disease']
for t in targets:
    if t not in df.columns:
        df[t] = 0
    else:
        df[t] = df[t].fillna(0)

Y = df[targets].copy()

# FULL feature list (only keep those present)
full_features = [
    'age', 'sex', 'BMI', 'systolic_bp', 'fasting_glucose',
    'resting_blood_pressure', 'cholestoral', 'fasting_blood_sugar',
    'rest_ecg', 'Max_heart_rate', 'exercise_induced_angina',
    'oldpeak', 'slope', 'vessels_colored_by_flourosopy',
    'thalassemia', 'chest_pain_type'
]
full_features = [f for f in full_features if f in df.columns]
print("Full features available:", full_features)

# Create per-target feature sets (obesity excludes BMI)
feature_sets = [
    full_features,                       # diabetes (uses BMI)
    [f for f in full_features if f != 'BMI'],  # obesity (NO BMI)
    full_features                        # heart
]

print("\\nFeature sets:")
print("diabetes:", feature_sets[0])
print("obesity (no BMI):", feature_sets[1])
print("heart:", feature_sets[2])

# Prepare train/test splits using the full dataset (we'll subset per pipeline)
X_all = df.copy()

X_train_all, X_test_all, Y_train, Y_test = train_test_split(
    X_all, Y, test_size=0.2, random_state=42
)

# Function to build a pipeline for a given feature set (uses only those columns)
def build_pipeline_for_features(df_sample, features):
    # Determine numeric vs categorical within these features
    df_sub = df_sample[features].copy()
    numeric = df_sub.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, categorical))
    preproc = ColumnTransformer(transformers) if transformers else None

    # build final pipeline
    if preproc is not None:
        pipe = Pipeline([
            ('preproc', preproc),
            ('rf', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced'))
        ])
    else:
        pipe = Pipeline([
            ('rf', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced'))
        ])
    return pipe

# Build one pipeline per target
pipelines = []
for feats in feature_sets:
    pipe = build_pipeline_for_features(X_train_all, feats)
    pipelines.append(pipe)

# Train each pipeline on its feature subset
for i, t in enumerate(targets):
    feats = feature_sets[i]
    pipe = pipelines[i]
    print(f"\\nTraining for target: {t} using {len(feats)} features")
    Xtr = X_train_all[feats]
    ytr = Y_train[t].astype(int)
    pipe.fit(Xtr, ytr)
    print(f" Trained pipeline for {t} (trained on {Xtr.shape[0]} rows)")

# Evaluate per-target on test subset
print("\\n=== Evaluation on test set (per target using corresponding features) ===")
for i, t in enumerate(targets):
    feats = feature_sets[i]
    pipe = pipelines[i]
    Xte = X_test_all[feats]
    yte = Y_test[t].astype(int)
    ypred = pipe.predict(Xte)
    print(f"\\n--- {t.upper()} ---")
    print(classification_report(yte, ypred, digits=3))
    # ROC AUC if possible
    try:
        if hasattr(pipe.named_steps['rf'], 'predict_proba'):
            probs = pipe.predict_proba(Xte)[:,1]
            print("ROC AUC:", roc_auc_score(yte, probs))
    except Exception:
        pass

# Save everything cleanly
out = {'pipelines': pipelines, 'feature_sets': feature_sets, 'targets': targets}
joblib.dump(out, 'multi_disease_model.joblib')
print("\\nSaved multi_disease_model.joblib")
