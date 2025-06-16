import streamlit as st



st.set_page_config(page_title="Homepage", layout="wide")

st.title("🎓 Student Performance Prediction")
st.markdown("""
Welcome to the **Student Prediction App**.

This app predicts student **marks** based on demographic information and then uses those marks to predict their **Pass/Fail** status.

Use the sidebar to:
- Explore the dataset
- View model training results
- Try out your own prediction
- See the conclusion and insights
""")
