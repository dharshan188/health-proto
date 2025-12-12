# train_quick.py
"""
Robust training script:
 - checks for features that are all-missing inside the training split and drops them
 - compatible with new scikit-learn (uses sparse_output=False)
 - trains RandomForest pipelines for diabetes, obesity, heart_disease
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_FP = 'data/merged.csv'
if not os.path.exists(DATA_FP):
    raise SystemExit("merged.csv not found. Run prepare_data.py first.")

df = pd.read_csv(DATA_FP)
print("Loaded merged.csv ->", df.shape)

# Candidate features
candidate = ['age', 'sex', 'BMI', 'systolic_bp', 'fasting_glucose']
features = [c for c in candidate if c in df.columns]

# initial categorical selection (global)
categorical_candidates = [c for c in features if df[c].dtype == object or c == "sex"]
categorical_global = [c for c in categorical_candidates if df[c].notna().any()]
numeric_global = [c for c in features if c not in categorical_global]

print("Initial numeric (global):", numeric_global)
print("Initial categorical (global):", categorical_global)

def build_preprocessor(numeric_feats, categorical_feats):
    transformers = []
    if numeric_feats:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipe, numeric_feats))
    if categorical_feats:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, categorical_feats))
    if not transformers:
        raise RuntimeError("No valid features after filtering.")
    return ColumnTransformer(transformers)

def train_target(target, filename):
    if target not in df.columns:
        print(f"[SKIP] target '{target}' not found.")
        return

    data = df[df[target].notna()].copy()
    nrows = len(data)
    if nrows < 50:
        print(f"[SKIP] Not enough samples for {target} ({nrows} rows).")
        return

    # Use the global lists but we'll filter by training split later
    X = data[numeric_global + categorical_global]
    y = data[target].astype(int)

    # stratify only if both classes present
    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Now remove any features that are entirely missing in X_train
    numeric_feats = [c for c in numeric_global if c in X_train.columns and X_train[c].notna().any()]
    categorical_feats = [c for c in categorical_global if c in X_train.columns and X_train[c].notna().any()]

    print(f"\n[TRAIN] {target}: train={len(X_train)}, test={len(X_test)}")
    print(" -> numeric used:", numeric_feats)
    print(" -> categorical used:", categorical_feats)

    if not numeric_feats and not categorical_feats:
        print(f"[SKIP] No usable features for {target} after filtering missing values.")
        return

    preproc = build_preprocessor(numeric_feats, categorical_feats)

    # Build pipeline and train
    clf = Pipeline([
        ('preproc', preproc),
        ('rf', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced'))
    ])

    clf.fit(X_train[numeric_feats + categorical_feats], y_train)

    y_pred = clf.predict(X_test[numeric_feats + categorical_feats])
    print(classification_report(y_test, y_pred, digits=3))

    # ROC AUC if applicable
    try:
        if len(set(y_test)) > 1 and hasattr(clf.named_steps['rf'], "predict_proba"):
            y_prob = clf.predict_proba(X_test[numeric_feats + categorical_feats])[:, 1]
            print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except Exception as e:
        print("Warning computing ROC AUC:", e)

    joblib.dump(clf, filename)
    print(f"[SAVED] {filename}")

# Train models
train_target('diabetes', 'diabetes_pipeline.joblib')
train_target('obesity', 'obesity_pipeline.joblib')
if 'heart_disease' in df.columns:
    train_target('heart_disease', 'heart_pipeline.joblib')
else:
    print("[SKIP] No heart_disease column present.")
