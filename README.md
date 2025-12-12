Health Proto – Multi-Disease Prediction Model

A machine learning system that predicts the risk of Diabetes, Obesity, and Heart Disease from clinical health features.
This project includes data preparation, multi-output model training, and single-sample prediction scripts.

 Features

✔ A single trained ML model that predicts 3 diseases

✔ Obesity prediction without using BMI

✔ Individual pipelines for each disease (for analysis)

✔ Scripts to train, test, and make predictions

✔ Clean probability outputs for each disease

✔ Lightweight and easy to run

Project Structure
health-proto/
│
├── data/
│   ├── pima.csv                 # Diabetes dataset
│   ├── obesity.csv              # Obesity dataset
│   ├── heart.csv                # Heart disease dataset
│   ├── merged.csv               # Combined dataset for multi-output model
│
├── multi_disease_model.joblib   # FINAL combined model (3-disease predictor)
├── diabetes_pipeline.joblib     # Diabetes-only model
├── obesity_no_bmi_pipeline.joblib
├── heart_pipeline.joblib
│
├── prepare_data.py              # Merges and cleans datasets
├── train_multi_output.py        # Trains the combined 3-disease model
├── train_quick.py               # Quick single-disease training
├── train_obesity_no_bmi.py      # Obesity training without BMI
│
├── predict_single.py            # Predicts for 1 user's input
├── predict_all.py               # Predicts for a CSV file of multiple users
│
└── README.md                    # You are here!

 Installation
1️) Clone the repository
git clone https://github.com/Gokul-pixel-1/health-proto.git
cd health-proto

2️)Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\activate    # Windows

3️) Install dependencies
pip install -r requirements.txt


If a requirements file is not included, install manually:

pip install pandas scikit-learn joblib

 Training the Combined 3-Disease Model

Run this to train the multi-output model:

python train_multi_output.py


This will generate:

multi_disease_model.joblib


This model predicts:

diabetes

obesity

heart_disease

using one input sample.

 Making Predictions (Single User)

Use:

python predict_single.py


Example output:

Predictions:
{'diabetes': 0, 'obesity': 0, 'heart_disease': 1}

Probabilities:
{'diabetes': 0.04, 'obesity': 0.11, 'heart_disease': 0.59}

Input Format for Predictions

Your input should include:

age

sex (M/F)

BMI

systolic_bp

fasting_glucose

resting_blood_pressure

cholestoral

fasting_blood_sugar (0/1)

rest_ecg

Max_heart_rate

exercise_induced_angina (0/1)

oldpeak

slope

vessels_colored_by_flourosopy

thalassemia

chest_pain_type

Example (inside predict_single.py):

sample = {
    "age": 45,
    "sex": "M",
    "BMI": 28.5,
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

Predicting Multiple Users (CSV Input)

Add your input rows to a CSV, then run:

python predict_all.py


Output CSV will include predictions for all users.

Contributing

Pull requests are welcome!
If you'd like to improve performance or add a Streamlit UI, feel free to contribute.

📜 License

This project is open-source and free to use.
