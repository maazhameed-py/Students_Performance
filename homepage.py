import streamlit as st

st.set_page_config(page_title="Student Prediction App", layout="wide")

st.title("\U0001F393 Student Performance Prediction Using Demographics → Marks → Pass/Fail")

st.header("Welcome to the Student Performance Predictor")
st.markdown("""
This app predicts student marks and pass/fail outcomes using demographic and behavioral data.

**Sections:**
- 📊 EDA: Explore the dataset
- 🧠 Modelling: Predict marks and pass/fail outcomes
- 🧪 Try it Yourself: Input your own data

> Use the sidebar to navigate between pages.
""")