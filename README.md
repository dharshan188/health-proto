# Health Proto – Multi-Disease Prediction Model

A machine learning system that predicts the risk of Diabetes, Obesity, and Heart Disease from clinical health features.
This project includes data preparation, multi-output model training, and an interactive web application.

## Features

✔ A single trained ML model that predicts 3 diseases

✔ An interactive Streamlit web application to get instant predictions

✔ Obesity prediction without using BMI

✔ Individual pipelines for each disease (for analysis)

✔ Scripts to train, test, and make predictions

✔ Clean probability outputs for each disease

✔ Lightweight and easy to run

## Project Structure
```
health-proto/
│
├── data/
│   ├── pima.csv
│   ├── obesity.csv
│   ├── heart.csv
│   └── merged.csv
│
├── app.py                       # NEW Streamlit web application
├── requirements.txt             # NEW dependencies file
│
├── multi_disease_model.joblib
├── diabetes_pipeline.joblib
├── obesity_no_bmi_pipeline.joblib
├── heart_pipeline.joblib
│
├── prepare_data.py
├── train_multi_output.py
├── train_quick.py
├── train_obesity_no_bmi.py
│
├── predict_single.py
├── predict_all.py
│
└── README.md
```

## Installation
1️) **Clone the repository**
```bash
git clone https://github.com/Gokul-pixel-1/health-proto.git
cd health-proto
```

2️) **Create and activate a virtual environment**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3️) **Install dependencies**
```bash
pip install -r requirements.txt
```

## Running the Web Application
To start the interactive prediction application, run:
```bash
streamlit run app.py
```
This will open a new tab in your browser with the application.

## Making Predictions (Command Line)

### Single User
Use:
```bash
python predict_single.py
```
**Example output:**
```
Predictions:
{'diabetes': 0, 'obesity': 0, 'heart_disease': 1}

Probabilities:
{'diabetes': 0.04, 'obesity': 0.11, 'heart_disease': 0.59}
```

### Multiple Users (CSV Input)
Add your input rows to a CSV, then run:
```bash
python predict_all.py
```
The output CSV will include predictions for all users.

## Training the Combined 3-Disease Model

Run this to train the multi-output model:
```bash
python train_multi_output.py
```
This will generate `multi_disease_model.joblib`.

## Contributing

Pull requests are welcome! If you'd like to improve performance, add new features, or enhance the UI, feel free to contribute.

## License

This project is open-source and free to use.
