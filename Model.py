import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report

st.set_page_config(page_title="Prediction Models", layout="wide")

st.title("🧠 Model: Predict Scores and Pass/Fail")

# Load Data
df = pd.read_csv("StudentsPerformance_dataset.csv")

features_demo = ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                 'test preparation course', 'parent_education_level', 'study_hours_per_week',
                 'attendance_rate', 'internet_access', 'school_support', 'past_failures']

le = LabelEncoder()
df_encoded = df.copy()
for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
            'test preparation course', 'internet_access', 'school_support']:
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded[features_demo]
y_math = df_encoded['math score']
y_reading = df_encoded['reading score']
y_writing = df_encoded['writing score']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_math_train, y_math_test = train_test_split(X_scaled, y_math, test_size=0.2, random_state=42)
_, _, y_reading_train, y_reading_test = train_test_split(X_scaled, y_reading, test_size=0.2, random_state=42)
_, _, y_writing_train, y_writing_test = train_test_split(X_scaled, y_writing, test_size=0.2, random_state=42)

math_model = RandomForestRegressor()
reading_model = RandomForestRegressor()
writing_model = RandomForestRegressor()

math_model.fit(X_train, y_math_train)
reading_model.fit(X_train, y_reading_train)
writing_model.fit(X_train, y_writing_train)

math_preds = math_model.predict(X_test)
reading_preds = reading_model.predict(X_test)
writing_preds = writing_model.predict(X_test)

st.subheader("Regression Performance (Demographics → Marks)")
st.write(f"✍️ MSE (Math): {mean_squared_error(y_math_test, math_preds):.2f}")
st.write(f"✍️ MSE (Reading): {mean_squared_error(y_reading_test, reading_preds):.2f}")
st.write(f"✍️ MSE (Writing): {mean_squared_error(y_writing_test, writing_preds):.2f}")

# Classifier Model
df_encoded['predicted_avg'] = df_encoded[['math score', 'reading score', 'writing score']].mean(axis=1)
df_encoded['target'] = df_encoded['target'].apply(lambda x: 1 if x == 'Pass' else 0)

marks_only = df_encoded[['math score', 'reading score', 'writing score']]
target = df_encoded['target']

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(marks_only, target, stratify=target, test_size=0.2, random_state=42)
clf_model = RandomForestClassifier()
clf_model.fit(X_train_m, y_train_m)

y_pred_m = clf_model.predict(X_test_m)
acc = accuracy_score(y_test_m, y_pred_m)

st.subheader("Classification Performance (Marks → Pass/Fail)")
st.write(f"✅ Accuracy: {acc:.2f}")
st.text("Classification Report:")
st.text(classification_report(y_test_m, y_pred_m))

# User Input
st.header("Try It Yourself")

def user_input_form():
    with st.form("user_input"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", df['gender'].unique())
            ethnicity = st.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
            parent_edu = st.selectbox("Parental Level of Education", df['parental level of education'].unique())
            lunch = st.selectbox("Lunch", df['lunch'].unique())
            prep = st.selectbox("Test Preparation Course", df['test preparation course'].unique())

        with col2:
            parent_lvl = st.slider("Parent Education Level (1-6)", 1, 6, 3)
            study_hours = st.slider("Study Hours/Week", 0, 40, 10)
            attendance = st.slider("Attendance Rate (%)", 50, 100, 80)
            past_fails = st.slider("Past Failures", 0, 3, 0)
            internet = st.selectbox("Internet Access", df['internet_access'].unique())
            support = st.selectbox("School Support", df['school_support'].unique())

        submitted = st.form_submit_button("Predict Outcome")
        if submitted:
            return {
                'gender': gender,
                'race/ethnicity': ethnicity,
                'parental level of education': parent_edu,
                'lunch': lunch,
                'test preparation course': prep,
                'parent_education_level': parent_lvl,
                'study_hours_per_week': study_hours,
                'attendance_rate': attendance,
                'past_failures': past_fails,
                'internet_access': internet,
                'school_support': support
            }
    return None

user_input = user_input_form()

if user_input:
    user_df = pd.DataFrame([user_input])
    for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch',
                'test preparation course', 'internet_access', 'school_support']:
        user_df[col] = le.fit_transform(user_df[col])

    user_scaled = scaler.transform(user_df[features_demo])

    math_pred = math_model.predict(user_scaled)[0]
    reading_pred = reading_model.predict(user_scaled)[0]
    writing_pred = writing_model.predict(user_scaled)[0]

    st.subheader("\U0001F4CA Predicted Marks")
    st.write(f"Math: {math_pred:.2f}")
    st.write(f"Reading: {reading_pred:.2f}")
    st.write(f"Writing: {writing_pred:.2f}")

    pred_df = pd.DataFrame([[math_pred, reading_pred, writing_pred]], columns=['math score', 'reading score', 'writing score'])
    pass_pred = clf_model.predict(pred_df)[0]
    result = "✅ Pass" if pass_pred == 1 else "❌ Fail"
    st.subheader("🎯 Final Prediction")
    st.success(f"The student is predicted to: **{result}**")