## Approach

1. **EDA** — explored distributions, correlations, and target-vs-feature relationships
   to understand which checkup variables are most informative.
2. **Feature engineering** — added `age_group` (binned age) and `oldpeak_slope`
   (interaction term between ST depression and ST slope) to capture non-linear
   clinical patterns.
3. **Model comparison** — trained and evaluated three distinct algorithms:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting

   Compared against a `DummyClassifier` baseline using accuracy, precision, recall,
   F1, and ROC-AUC.
4. **Metric priority** — **recall** on the disease class was prioritised, since
   missing a true heart-disease case (false negative) is clinically far more costly
   than a false alarm.
5. **Hyperparameter tuning** — `RandomizedSearchCV` (max 3 values per hyperparameter)
   on the best-performing model, with before/after comparison against the untuned
   baseline.
6. **Deployment** — the final tuned pipeline is served through an interactive
   Streamlit web app (`app.py`) with live predictions as inputs change.

## Repository Contents

| File | Description |
|---|---|
| `MLDP_Program_Codes_2502140C.ipynb` | Full CRISP-DM notebook: EDA, preprocessing, model training, tuning, evaluation |
| `app.py` | Streamlit web app — loads the trained pipeline and serves live predictions |
| `requirements.txt` | Python dependencies for running/deploying the app |
| `heart_disease_model.joblib` | Final tuned scikit-learn pipeline (saved from the notebook) |

## Running Locally

pip install -r requirements.txt
python -m streamlit run app.py


