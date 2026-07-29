"""
SmartCare Cardiac Risk Screener
Streamlit web app for CAI2C08 MLDP project — heart disease risk prediction
Loads the trained pipeline exported from the Jupyter notebook.
"""

import streamlit as st
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Page config & light styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartCare Cardiac Risk Screener",
    page_icon="🫀",
    layout="centered",
)

st.markdown("""
<style>
.big-font { font-size:16px !important; }
.risk-high { background-color:#fdecea; padding:20px; border-radius:10px; border:1px solid #f5c2c0; }
.risk-low { background-color:#eafaf1; padding:20px; border-radius:10px; border:1px solid #b7ebc6; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load model (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.joblib")
    feature_cols = joblib.load("model_feature_columns.joblib")
    return model, feature_cols


model, feature_cols = load_model()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🫀 SmartCare Cardiac Risk Screener")
st.markdown(
    "A clinical decision-support tool that estimates a patient's risk of heart "
    "disease from routine checkup data, helping prioritise who should be sent "
    "for further cardiac testing (e.g. angiography)."
)
st.divider()

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.subheader("Patient checkup data")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 54)
        sex = st.radio("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female", horizontal=True)
        cp = st.selectbox(
            "Chest pain type", options=[1, 2, 3, 4],
            format_func=lambda x: {1: "Typical angina", 2: "Atypical angina",
                                    3: "Non-anginal pain", 4: "Asymptomatic"}[x],
        )
        trestbps = st.slider("Resting blood pressure (mm Hg)", 80, 220, 130)
        chol = st.slider("Serum cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.radio("Fasting blood sugar > 120 mg/dl?", options=[1, 0],
                       format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        restecg = st.selectbox(
            "Resting ECG result", options=[0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality",
                                    2: "Left ventricular hypertrophy"}[x],
        )

    with col2:
        thalach = st.slider("Maximum heart rate achieved", 60, 220, 150)
        exang = st.radio("Exercise-induced angina?", options=[1, 0],
                          format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
        oldpeak = st.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
        slope = st.selectbox(
            "Slope of peak exercise ST segment", options=[1, 2, 3],
            format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
        )
        ca = st.selectbox("Number of major vessels coloured by fluoroscopy (0-3)", options=[0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia test result", options=[3, 6, 7],
            format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}[x],
        )

    submitted = st.form_submit_button("Predict risk", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submitted:
    raw = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": float(ca), "thal": float(thal),
    }])

    # Recreate the same engineered features used during training
    raw["age_group"] = pd.cut(raw["age"], bins=[0, 45, 55, 65, 100],
                               labels=["<=45", "46-55", "56-65", "65+"])
    raw["oldpeak_slope"] = raw["oldpeak"] * raw["slope"]

    raw = raw[feature_cols]  # ensure exact column order used at training time

    pred = model.predict(raw)[0]
    proba = model.predict_proba(raw)[0][1]

    st.divider()
    st.subheader("Result")

    if pred == 1:
        st.markdown(
            f"<div class='risk-high'><h3>⚠️ High risk</h3>"
            f"<p class='big-font'>Estimated probability of heart disease: <b>{proba:.0%}</b></p>"
            f"<p class='big-font'>Recommendation: prioritise this patient for further cardiac testing "
            f"(e.g. angiography).</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='risk-low'><h3>✅ Low risk</h3>"
            f"<p class='big-font'>Estimated probability of heart disease: <b>{proba:.0%}</b></p>"
            f"<p class='big-font'>Recommendation: routine follow-up; further testing not immediately "
            f"indicated based on this data.</p></div>",
            unsafe_allow_html=True,
        )


st.divider()
with st.expander("About this tool"):
    st.write(
        "Model: trained and tuned in the accompanying Jupyter notebook using the UCI "
        "Heart Disease dataset (Cleveland Clinic Foundation subset, 303 patients). "
        "Priority evaluation metric: recall on the disease class, since missing a "
        "true heart-disease case is more costly than a false alarm in this clinical "
        "screening context."
    )
