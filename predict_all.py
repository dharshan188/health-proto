# predict_all.py
# Loads the multi_disease_model and runs a single-sample test prediction.
import joblib
import pandas as pd
import numpy as np

# load model
model = joblib.load("multi_disease_model.joblib")

# try to infer which input feature names the preprocessor expects
preproc = model.named_steps['preproc']
feature_names = []

# ColumnTransformer: get original column lists from transformers
for name, trans, cols in preproc.transformers_:
    # cols might be a list of names
    if hasattr(cols, '__iter__'):
        feature_names.extend(list(cols))
    else:
        feature_names.append(cols)

# Build a sample row using sensible defaults for numeric and strings for categorical
sample = {}
for f in feature_names:
    # cheap heuristic defaults
    if f.lower() in ('age',):
        sample[f] = 45
    elif 'bp' in f.lower() or 'pressure' in f.lower() or 'chol' in f.lower() or 'max' in f.lower() or 'fasting' in f.lower() or 'glucose' in f.lower() or 'oldpeak' in f.lower():
        sample[f] = 120  # numeric default
    elif f.lower() in ('bmi',):
        sample[f] = 28.5
    else:
        # categorical/text default
        sample[f] = "M"

# Create DataFrame with one row
X = pd.DataFrame([sample], columns=feature_names)

print("Using input features:", feature_names)
print("\nSample input row:")
print(X.to_string(index=False))

# Predict
pred = model.predict(X)
print("\nPredictions (diabetes, obesity, heart_disease):", pred)

# Try to get probability per target (may be None for some targets)
probs = model.named_steps['clf'].predict_proba(preproc.transform(X))  # returns list per target
print("\nPer-target probability arrays (None if not available):")
for i,p in enumerate(probs):
    print(f"Target {i} probs shape:", None if p is None else p.shape)
