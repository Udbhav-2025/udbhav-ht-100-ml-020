

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt

MODEL_PATH = "cardioastra_model.joblib"

st.set_page_config(page_title="CardioAstra — Cardio Predictor", layout="wide")

# ---------- Utility: create synthetic demo data (only for MVP / demo) ----------
def create_synthetic_data(n=5000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(20, 85, size=n)                      # years
    sex = rng.binomial(1, 0.48, size=n)                   # 0 female, 1 male
    bmi = rng.normal(27, 5, size=n)                       # kg/m2
    systolic = rng.normal(130, 18, size=n)                # mmHg
    diastolic = rng.normal(80, 10, size=n)                # mmHg
    cholesterol = rng.normal(200, 40, size=n)             # mg/dL
    smoking = rng.binomial(1, 0.2, size=n)
    diabetes = rng.binomial(1, 0.12, size=n)
    family = rng.binomial(1, 0.18, size=n)
    activity = rng.exponential(1.5, size=n)               # hours/week

    # Create risk probability using a logistic combination (for synthetic labels)
    linear = (
        0.04*(age - 50) +
        0.8*sex +
        0.06*(bmi - 25) +
        0.03*(systolic - 120) +
        0.02*(cholesterol - 180) +
        0.9*smoking +
        1.1*diabetes +
        0.7*family -
        0.15*(activity)  # more activity lowers risk
    )
    prob = 1 / (1 + np.exp(-linear))
    # create labels using thresholded probability (noisy)
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
    # Add some missingness to emulate real-world data
    for col in ["bmi", "cholesterol", "systolic_bp", "physical_activity"]:
        mask = rng.rand(n) < 0.06
        df.loc[mask, col] = np.nan
    return df

# ---------- Train or load model ----------
def get_or_train_model(force_retrain=False):
    if os.path.exists(MODEL_PATH) and not force_retrain:
        model_bundle = joblib.load(MODEL_PATH)
        return model_bundle
    # Train pipeline: imputer -> scaler -> logistic regression
    df = create_synthetic_data()
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    # We'll fit imputer and scaler on training set and store them with classifier
    X_train_imp = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    X_test_imp = imputer.transform(X_test)
    X_test_scaled = scaler.transform(X_test_imp)
    preds = clf.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, preds)

    model_bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "clf": clf,
        "feature_names": list(X.columns),
        "auc": auc
    }
    joblib.dump(model_bundle, MODEL_PATH)…
