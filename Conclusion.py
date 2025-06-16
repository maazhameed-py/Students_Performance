import streamlit as st



st.set_page_config(page_title="Conclusion", layout="wide")
st.title("📌 Conclusion")
st.write("This is the Conclusion page.")



st.markdown("""
This project followed a two-step approach:
1. **Demographics to Marks Prediction**:  
   We used student demographic and support-related features to predict scores in Math, Reading, and Writing.

2. **Marks to Outcome Prediction**:  
   Based on those scores, we predicted whether the student would **Pass or Fail**.

---

### 🔍 Key Takeaways:
- **Study hours** and **attendance** are crucial to academic performance.
- **Parental involvement** and **school support** also influence student outcomes.
- The models achieved solid performance with meaningful predictions.

---

### 🧪 What You Can Do:
Use the app interactively to:
- Adjust inputs such as study hours, parental education, or attendance.
- Observe how different factors impact student outcomes.

---

This showcases how machine learning can help educators and stakeholders **predict**, **understand**, and **intervene** effectively to support student success.
""")
