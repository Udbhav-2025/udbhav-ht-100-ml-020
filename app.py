
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import os

MODEL_PATH = "cardioastra_model.joblib"

st.set_page_config(page_title="CardioAstra — Cardio Predictor", layout="wide")

# ---------- Utility: create synthetic demo data ----------
def create_synthetic_data(n=5000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(20, 85, size=n)
    sex = rng.binomial(1, 0.48, size=n)
    bmi = rng.normal(27, 5, size=n)
    systolic = rng.normal(130, 18, size=n)
    diastolic = rng.normal(80, 10, size=n)
    cholesterol = rng.normal(200, 40, size=n)
    smoking = rng.binomial(1, 0.2, size=n)
    diabetes = rng.binomial(1, 0.12, size=n)
    family = rng.binomial(1, 0.18, size=n)
    activity = rng.exponential(1.5, size=n)

    linear = (
        0.04*(age - 50) +
        0.8*sex +
        0.06*(bmi - 25) +
        0.03*(systolic - 120) +
        0.02*(cholesterol - 180) +
        0.9*smoking +
        1.1*diabetes +
        0.7*family -
        0.15*(activity)
    )
    prob = 1 / (1 + np.exp(-linear))
    labels = rng.binomial(1, prob)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "cholesterol": cholesterol,
        "smoking": smoking,
        "diabetes": diabetes,
        "family_history": family,
        "physical_activity": activity,
        "target": labels
    })

    for col in ["bmi", "cholesterol", "systolic_bp", "physical_activity"]:
        mask = rng.rand(n) < 0.06
        df.loc[mask, col] = np.nan

    return df

# ---------- Train or load model ----------
def get_or_train_model(force_retrain=False):
    if os.path.exists(MODEL_PATH) and not force_retrain:
        return joblib.load(MODEL_PATH)

    df = create_synthetic_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    clf.fit(X_train_scaled, y_train)

    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test_scaled)[:, 1])

    model_bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "clf": clf,
        "feature_names": list(X.columns),
        "auc": auc
    }

    joblib.dump(model_bundle, MODEL_PATH)
    return model_bundle

# ---------- Load or train model ----------
model_bundle = get_or_train_model()

st.title("CardioAstra — Cardio Predictor")
st.write(f"Model trained on synthetic data — AUC: {model_bundle['auc']:.2f}")

# ---------- Input form ----------
st.subheader("Enter your details:")

with st.form("input_form"):
    age = st.number_input("Age (years)", 20, 100, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    bmi = st.number_input("BMI (kg/m²)", 10.0, 50.0, 27.0)
    systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 130)
    diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 130, 80)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    family_history = st.selectbox("Family History of CVD", ["No", "Yes"])
    physical_activity = st.number_input("Physical Activity (hrs/week)", 0.0, 20.0, 1.0)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    X_input = pd.DataFrame([{
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "cholesterol": cholesterol,
        "smoking": 1 if smoking == "Yes" else 0,
        "diabetes": 1 if diabetes == "Yes" else 0,
        "family_history": 1 if family_history == "Yes" else 0,
        "physical_activity": physical_activity
    }])

    X_input_imp = model_bundle["imputer"].transform(X_input)
    X_input_scaled = model_bundle["scaler"].transform(X_input_imp)
    risk_prob = model_bundle["clf"].predict_proba(X_input_scaled)[:, 1][0]

    st.subheader("Predicted Cardiovascular Risk")
    st.write(f"{risk_prob*100:.2f}% chance of cardiovascular risk")

