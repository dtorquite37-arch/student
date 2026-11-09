# Student Performance Prediction System

A classroom prototype that estimates whether a student is **at risk of failing the final grade (G3)**, built for the Data Mining laboratory (Python, Streamlit, GitHub).

## Problem
Academic risk is often noticed only after poor results. This prototype explores early identification of students who may need additional academic support, using historical student indicators.

## Dataset
UCI Machine Learning Repository, *Student Performance* (ID 320):
https://archive.ics.uci.edu/dataset/320/student+performance
Retrieved automatically with the `ucimlrepo` package.

## Prediction target
- `G3 < 10` -> **AT RISK** (1)
- `G3 >= 10` -> **NOT AT RISK** (0)

`G3` is used only to create the target and is **never** an input feature (prevents target leakage).

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest), joblib, Streamlit, ucimlrepo, Git/GitHub.

## Installation and run
```bash
pip install -r requirements.txt
python train_model.py      # downloads data, trains model, creates student_data.csv and student_model.pkl
streamlit run app.py       # opens http://localhost:8501
```
In the app, download the sample input CSV, then upload it to see predictions. Results can be downloaded as CSV.

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalize to other institutions or populations.
- G1 and G2 (earlier grades) are strong predictors, so the model is mostly reflecting prior performance.
- Predictions are **not diagnoses** and must not be the sole basis for real student decisions.
