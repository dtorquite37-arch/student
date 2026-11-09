# app.py
import subprocess
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Predict", page_icon="🎓", layout="wide")


# Files are looked up next to app.py, no matter where Streamlit was started
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"
DATA_PATH = BASE_DIR / "student_data.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# If train_model.py has not been run yet, offer to run it from here
if not (MODEL_PATH.exists() and DATA_PATH.exists()):
    st.error(
        "student_model.pkl or student_data.csv not found. "
        "Run `python train_model.py` first, or click the button below."
    )
    if st.button("Run train_model.py now (needs internet)"):
        with st.spinner("Downloading dataset and training the model..."):
            result = subprocess.run(
                [sys.executable, str(BASE_DIR / "train_model.py")],
                cwd=str(BASE_DIR), capture_output=True, text=True
            )
        if result.returncode == 0 and MODEL_PATH.exists() and DATA_PATH.exists():
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Training failed. Details below:")
            st.code(result.stdout + "\n" + result.stderr)
    st.stop()

model = load_model()
data = pd.read_csv(DATA_PATH)

st.title("🎓 Student Performance Prediction")
st.caption("Classroom prototype for academic risk exploration")
st.info(
    "Responsible use: predictions are not diagnoses and must not be used "
    "as the sole basis for real student decisions."
)

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average final grade", round(data["G3"].mean(), 2))
c3.metric("Historical at-risk records", int((data["G3"] < 10).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10))

st.subheader("Predict uploaded student records")
st.write("Upload a CSV containing the same predictor columns as the training data.")

template = data.drop(columns=["G3"]).head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_prediction_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose student CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the uploaded file as CSV: {e}")
        st.stop()

    required_columns = list(model.feature_names_in_)
    missing = [c for c in required_columns if c not in new_data.columns]

    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
    else:
        try:
            predictions = model.predict(new_data[required_columns])
        except Exception as e:
            st.error(f"Prediction failed. Please check the values in your file: {e}")
            st.stop()

        results = new_data.copy()
        results["Prediction"] = [
            "AT RISK" if value == 1 else "NOT AT RISK"
            for value in predictions
        ]

        st.subheader("Prediction results")
        st.dataframe(results)
        st.metric("Predicted at-risk records", int((predictions == 1).sum()))

        st.download_button(
            "Download prediction results",
            results.to_csv(index=False),
            "student_predictions.csv",
            "text/csv"
        )
