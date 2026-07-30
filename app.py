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
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --ink: #0B1F3A;
    --ink-soft: #475467;
    --paper: #EEF2F8;
    --line: #E4E7EC;
    --accent: #2563EB;
    --accent-soft: #DBEAFE;
    --risk-high-bg: #FEE2E2;
    --risk-high-line: #F87171;
    --risk-high-text: #B91C1C;
    --risk-low-bg: #DCFCE7;
    --risk-low-line: #4ADE80;
    --risk-low-text: #15803D;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: var(--paper); }

/* ---- Instrument header bar ---- */
.instrument-header {
    background: linear-gradient(90deg, #1E3A8A 0%, #2563EB 50%, #38BDF8 100%);
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 8px 24px rgba(29, 78, 216, 0.3);
}
.instrument-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.01em;
}
.instrument-header p {
    color: #DBEAFE;
    margin: 0.35rem 0 0 0;
    font-size: 0.92rem;
    max-width: 560px;
}
.live-badge {
    background: rgba(20, 184, 166, 0.18);
    border: 1px solid #2DD4BF;
    color: #5EEAD4;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    white-space: nowrap;
}
.live-dot {
    display: inline-block;
    width: 6px; height: 6px;
    background: #2DD4BF;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 1.6s infinite;
}
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

/* ---- Panel section labels/titles (used inside real st.container(border=True)) ---- */
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.panel-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 0.9rem;
}

/* ---- Risk readout ---- */
.risk-high, .risk-low {
    padding: 1.4rem 1.6rem;
    border-radius: 12px;
    border-width: 1px;
    border-style: solid;
}
.risk-high { background-color: var(--risk-high-bg); border-color: var(--risk-high-line); }
.risk-low  { background-color: var(--risk-low-bg);  border-color: var(--risk-low-line); }
.risk-high h3 { color: var(--risk-high-text); margin: 0 0 0.4rem 0; font-size: 1.15rem; }
.risk-low h3  { color: var(--risk-low-text);  margin: 0 0 0.4rem 0; font-size: 1.15rem; }
.risk-figure {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    font-size: 2.2rem;
    color: var(--ink);
    line-height: 1;
    margin: 0.2rem 0 0.6rem 0;
}
.risk-sub { color: var(--ink-soft); font-size: 0.88rem; margin: 0; }

/* ---- Gauge bar ---- */
.gauge-track {
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: var(--line);
    margin: 0.9rem 0 0.4rem 0;
    overflow: visible;
}
.gauge-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #16A34A, #FACC15 55%, #DC2626);
}
.gauge-marker {
    position: absolute;
    top: -5px;
    width: 3px; height: 20px;
    background: var(--ink);
    border-radius: 2px;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--ink-soft);
    margin-top: 0.2rem;
}

/* ---- Streamlit's real bordered containers (replaces the old div hack) ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(16, 24, 40, 0.06);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-radius: 16px !important;
}

/* ---- Streamlit widget refinements ---- */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; color: var(--ink); }
div[data-testid="stExpander"] {
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fff;
}
.stSlider label, .stRadio label, .stSelectbox label { color: var(--ink); font-weight: 500; }
hr { border-color: var(--line) !important; }

/* ---- Slider color override (CSS-only, no theme config file) ---- */
div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--accent) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
    background: var(--line) !important;
}

/* ---- Dropdowns (selectboxes) — make them clearly stand out as clickable ---- */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    border: 2px solid var(--accent) !important;
    border-radius: 10px !important;
    background: var(--accent-soft) !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15);
    transition: box-shadow 0.15s ease;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
}
[data-testid="stSelectbox"] svg { fill: var(--accent) !important; }

</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Load model (cached so it only loads once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("heart_disease_model.joblib")
    return model


model = load_model()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="instrument-header">
    <div>
        <h1>🫀 SmartCare Cardiac Risk Screener</h1>
        <p>Clinical decision-support tool estimating a patient's heart disease risk
        from routine checkup data, to help prioritise who should be sent for
        further cardiac testing.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Two-column page layout: input widgets on the left, results panel on the right
# NOTE: no st.form here on purpose — plain widgets rerun the script (and update
# the prediction) the instant any value changes, giving live updates.
# ----------------------------------------------------------------------------
input_col, result_col = st.columns([1.1, 1], gap="large")

with input_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Patient checkup data</div>', unsafe_allow_html=True)

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

        # --------------------------------------------------------------------
        # Glossary — brief explanation of what each field means, for a
        # non-clinical audience (e.g. the lecturer/investor persona) viewing
        # the demo
        # --------------------------------------------------------------------
        with st.expander("ℹ️ What do these fields mean?"):
            st.markdown(
                """
                | Field | What it means |
                |---|---|
                | **Age** | Patient's age in years. |
                | **Sex** | Biological sex of the patient. |
                | **Chest pain type** | The nature of chest pain reported — asymptomatic is often *more* concerning clinically, since disease can be present without typical pain. |
                | **Resting blood pressure** | Blood pressure (mm Hg) measured at rest, on hospital admission. |
                | **Serum cholesterol** | Cholesterol level in the blood (mg/dl); a standard cardiovascular risk marker. |
                | **Fasting blood sugar** | Whether blood sugar after fasting exceeds 120 mg/dl — a diabetes-related risk indicator. |
                | **Resting ECG result** | Electrocardiogram reading at rest; abnormalities can indicate heart strain or damage. |
                | **Maximum heart rate achieved** | Highest heart rate reached during an exercise stress test — lower peaks can signal reduced cardiac capacity. |
                | **Exercise-induced angina** | Whether exercise brings on chest pain — a direct sign of restricted blood flow under stress. |
                | **ST depression (oldpeak)** | How much the heart's electrical ST segment dips during exercise vs. rest; larger dips suggest more ischemia (reduced blood flow). |
                | **Slope of peak exercise ST segment** | The shape of the ST segment at peak exercise; a downsloping pattern is generally a stronger warning sign than upsloping. |
                | **Major vessels coloured by fluoroscopy** | Number of major blood vessels (0–3) visibly narrowed on a fluoroscopy scan — more narrowed vessels usually means more severe disease. |
                | **Thalassemia test result** | Result of a blood-flow imaging test; "fixed" or "reversible defect" results indicate abnormal blood flow patterns linked to heart disease. |
                """
            )

# ----------------------------------------------------------------------------
# Prediction — recomputed on every rerun (i.e. every time any widget changes),
# so the result panel updates live without needing a submit button
# ----------------------------------------------------------------------------
raw = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": float(ca), "thal": float(thal),
}])

# Recreate the same engineered features used during training
raw["age_group"] = pd.cut(raw["age"], bins=[0, 45, 55, 65, 100],
                           labels=["<=45", "46-55", "56-65", "65+"])
raw["oldpeak_slope"] = raw["oldpeak"] * raw["slope"]

# NOTE: no column reordering needed here — the trained pipeline's
# ColumnTransformer selects columns by NAME (see notebook cell defining
# numeric_cols / categorical_cols), so column order in this DataFrame
# doesn't matter as long as all the right column names are present.

pred = model.predict(raw)[0]
proba = model.predict_proba(raw)[0][1]

# ----------------------------------------------------------------------------
# Right column — live results panel
# ----------------------------------------------------------------------------
gauge_pct = max(0.0, min(1.0, proba)) * 100

with result_col:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Risk assessment</div>', unsafe_allow_html=True)

        if pred == 1:
            st.markdown(
                f"""
                <div class='risk-high'>
                    <h3>⚠️ High risk</h3>
                    <div class="risk-figure">{proba:.0%}</div>
                    <p class="risk-sub">Estimated probability of heart disease.
                    Recommend prioritising this patient for further cardiac testing (e.g. angiography).</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class='risk-low'>
                    <h3>✅ Low risk</h3>
                    <div class="risk-figure">{proba:.0%}</div>
                    <p class="risk-sub">Estimated probability of heart disease.
                    Routine follow-up; further testing not immediately indicated based on this data.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="gauge-track">
                <div class="gauge-fill" style="width:{gauge_pct:.1f}%"></div>
                <div class="gauge-marker" style="left:calc({gauge_pct:.1f}% - 1px)"></div>
            </div>
            <div class="gauge-labels"><span>0% RISK</span><span>50% THRESHOLD</span><span>100% RISK</span></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="panel-label">ENTERED DATA</div>', unsafe_allow_html=True)
        st.dataframe(raw.T.rename(columns={0: "Value"}), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("About this tool"):
    st.write(
        "Model: trained and tuned in the accompanying Jupyter notebook using the UCI "
        "Heart Disease dataset (Cleveland Clinic Foundation subset, 303 patients). "
        "Priority evaluation metric: recall on the disease class, since missing a "
        "true heart-disease case is more costly than a false alarm in this clinical "
        "screening context."
    )
