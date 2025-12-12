# predict_single.py
import joblib
import pandas as pd

obj = joblib.load('multi_disease_model.joblib')
pipelines = obj['pipelines']
feature_sets = obj['feature_sets']
targets = obj['targets']

# Build one sample: fill only features used by pipelines
sample = {
  # numeric examples - change values to test different cases
  "age": 45,
  "sex": "M",
  "BMI": 28.5,                # present but obesity pipeline will ignore it
  "systolic_bp": 130,
  "fasting_glucose": 110,
  "resting_blood_pressure": 140,
  "cholestoral": 230,
  "fasting_blood_sugar": 0,
  "rest_ecg": "normal",
  "Max_heart_rate": 150,
  "exercise_induced_angina": 0,
  "oldpeak": 1.2,
  "slope": "flat",
  "vessels_colored_by_flourosopy": 0,
  "thalassemia": "normal",
  "chest_pain_type": "typical"
}

# build DataFrame with union of all feature names
all_feats = []
for fset in feature_sets:
    for f in fset:
        if f not in all_feats:
            all_feats.append(f)

# ensure sample has all features (use defaults if missing)
row = {f: sample.get(f, (45 if f=='age' else 'M')) for f in all_feats}
X = pd.DataFrame([row], columns=all_feats)

print("Input:")
print(X.to_string(index=False))

preds = {}
probs = {}
for i, target in enumerate(targets):
    feats = feature_sets[i]
    pipe = pipelines[i]
    Xsub = X[feats]
    preds[target] = int(pipe.predict(Xsub)[0])
    try:
        probs[target] = float(pipe.predict_proba(Xsub)[:,1][0])
    except Exception:
        probs[target] = None

print("\nPredictions (0=no, 1=yes):")
print(preds)
print("\nProbabilities (if available):")
print(probs)
