# SmartCare Cardiac Risk Screener — CAI2C08 MLDP Submission

**Student:** Tee Jia Zhun Louis (2502140C)

## Files
- `MLDP_Program_Codes_2502140C.ipynb` — full CRISP-DM notebook (EDA, cleaning, modelling, tuning, evaluation)
- `app.py` — Streamlit web app (loads the trained model and serves predictions)
- `heart_disease.csv` — cleaned Cleveland heart disease dataset (from UCI, with column headers added)
- `heart_disease_model.joblib` / `model_feature_columns.joblib` — exported trained pipeline used by app.py
- `requirements.txt` — dependencies for running the Streamlit app

## To run the app locally
```
pip install -r requirements.txt
streamlit run app.py
```
(Run this from a folder that also contains `heart_disease_model.joblib` and `model_feature_columns.joblib`.)

## To deploy
Push this folder to a GitHub repo, then deploy on Streamlit Community Cloud (share.streamlit.io) pointing at `app.py`. Once deployed, take the 3 required screenshots (before selection / after prediction / after changing inputs) for the Word document.
