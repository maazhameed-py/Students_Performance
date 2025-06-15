# 🎓 Student Performance Prediction App

This is a Streamlit-based machine learning application that predicts a student's academic performance and pass/fail status using demographic and educational support features. It mimics a two-step real-world decision model:  
**Demographics ➝ Marks ➝ Pass/Fail**

---

## 🚀 Features

### 1. Exploratory Data Analysis (EDA)
- Summary statistics
- Feature distribution plots (histograms, box plots)
- Correlation heatmaps
- Outlier detection
- Missing value analysis
- Grouped aggregations
- Pairwise relationships using seaborn heatmaps and scatter plots

### 2. Predictive Modeling
- **Step 1:** Predict scores in Math, Reading, and Writing using demographic/support features via **Random Forest Regressors**.
- **Step 2:** Predict **Pass/Fail** outcome from predicted marks using a **Random Forest Classifier**.

### 3. Interactive Student Outcome Predictor
- Enter custom student demographic and support data
- View predicted subject-wise marks
- View final **Pass/Fail** prediction

---

## 🧠 Model Workflow

```text
[ Demographics + Study Factors ]
             |
             v
   [Predicted Marks: Math, Reading, Writing]
             |
             v
        [Pass / Fail Prediction]
