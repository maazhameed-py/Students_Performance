import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




st.set_page_config(page_title="EDA", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.write("This is the EDA page.")


df = pd.read_csv("StudentsPerformance_dataset.csv")

st.subheader("Dataset Snapshot")
st.write(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())
st.write("Median:")
st.write(df.median(numeric_only=True))
st.write("Mode:")
st.write(df.mode().iloc[0])

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Target Balance")
st.bar_chart(df["target"].value_counts())

if 'average_score' not in df.columns:
    df['average_score'] = df[['math score', 'reading score', 'writing score']].mean(axis=1)

if 'score_variance' not in df.columns:
    df['score_variance'] = df[['math score', 'reading score', 'writing score']].var(axis=1)

if 'risk_factor' not in df.columns:
    df['risk_factor'] = (100 - df['attendance_rate']) + (3 - df['study_hours_per_week'] / 10)

st.subheader("Boxplots (Outlier Detection)")
for col in ['math score', 'reading score', 'writing score']:
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x=col, ax=ax)
    st.pyplot(fig)

st.subheader("Histograms")
for col in ['math score', 'reading score', 'writing score', 'study_hours_per_week', 'attendance_rate']:
    fig, ax = plt.subplots()
    sns.histplot(df[col], kde=True, ax=ax)
    st.pyplot(fig)

st.subheader("Grouped Means by Gender")
st.write(df.groupby('gender')[['math score', 'reading score', 'writing score']].mean())

st.subheader("Study Hours vs Scores")
for col in ['math score', 'reading score', 'writing score']:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='study_hours_per_week', y=col, ax=ax)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
num_cols = ['math score', 'reading score', 'writing score', 'average_score', 'score_variance',
            'study_hours_per_week', 'attendance_rate', 'parent_education_level', 'past_failures', 'risk_factor']
fig, ax = plt.subplots()
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)
