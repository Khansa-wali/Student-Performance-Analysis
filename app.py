import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Student Performance Intelligence System",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1100px;
}

.main-title{
    font-size:42px;
    font-weight:800;
    text-align:center;
    color:#1f4e79;
}

.sub-title{
    text-align:center;
    color:#666;
    margin-bottom:25px;
}

.footer-box{
    text-align:center;
    color:#777;
    font-size:14px;
    margin-top:30px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
reg_model = joblib.load("linear_regression_model.pkl")
pass_model = joblib.load("pass_fail_model.pkl")
ordinal_encoder = joblib.load("ordinal_encoder.pkl")
feature = joblib.load("features.pkl")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="main-title">🎓 AI Student Performance Intelligence System</div>
<div class="sub-title">Predict Exam Score, Pass/Fail Status and Insights</div>
""", unsafe_allow_html=True)

st.info("AI system to predict student performance using ML models.")
st.divider()

# -----------------------------
# INPUTS (SIDEBAR)
# -----------------------------
st.sidebar.header("📝 Student Inputs")

hours = st.sidebar.slider("Hours Studied", 0, 50, 10)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 80)
sleep = st.sidebar.slider("Sleep Hours", 0, 12, 7)
previous = st.sidebar.slider("Previous Scores", 0, 100, 70)
tutoring = st.sidebar.slider("Tutoring Sessions", 0, 10, 2)
physical = st.sidebar.slider("Physical Activity", 0, 20, 3)

parental = st.sidebar.selectbox("Parental Involvement", ["Low", "Medium", "High"])
resources = st.sidebar.selectbox("Access to Resources", ["Low", "Medium", "High"])
motivation = st.sidebar.selectbox("Motivation Level", ["Low", "Medium", "High"])
income = st.sidebar.selectbox("Family Income", ["Low", "Medium", "High"])
teacher = st.sidebar.selectbox("Teacher Quality", ["Low", "Medium", "High"])
peer = st.sidebar.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
parent_edu = st.sidebar.selectbox("Parental Education", ["High School", "College", "Postgraduate"])
distance = st.sidebar.selectbox("Distance from Home", ["Near", "Moderate", "Far"])

gender = st.sidebar.radio("Gender", ["Female", "Male"])
internet = st.sidebar.checkbox("Internet Access")
extra = st.sidebar.checkbox("Extracurricular Activities")
learning = st.sidebar.checkbox("Learning Disabilities")
school = st.sidebar.radio("School Type", ["Private", "Public"])

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🚀 Predict Performance"):

    # -----------------------------
    # ORDINAL ENCODING (FIXED SCOPE)
    # -----------------------------
    ordinal_input = pd.DataFrame([{
        "Parental_Involvement": parental,
        "Access_to_Resources": resources,
        "Motivation_Level": motivation,
        "Family_Income": income,
        "Teacher_Quality": teacher,
        "Peer_Influence": peer,
        "Parental_Education_Level": parent_edu,
        "Distance_from_Home": distance
    }])

    ordinal_values = ordinal_encoder.transform(ordinal_input)[0]

    # -----------------------------
    # FINAL INPUT DATAFRAME
    # -----------------------------
    input_df = pd.DataFrame([{
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Parental_Involvement": ordinal_values[0],
        "Access_to_Resources": ordinal_values[1],
        "Sleep_Hours": sleep,
        "Previous_Scores": previous,
        "Motivation_Level": ordinal_values[2],
        "Tutoring_Sessions": tutoring,
        "Family_Income": ordinal_values[3],
        "Teacher_Quality": ordinal_values[4],
        "Peer_Influence": ordinal_values[5],
        "Physical_Activity": physical,
        "Parental_Education_Level": ordinal_values[6],
        "Distance_from_Home": ordinal_values[7],
        "Gender_Male": 1 if gender == "Male" else 0,
        "Internet_Access_Yes": int(internet),
        "Extracurricular_Activities_Yes": int(extra),
        "Learning_Disabilities_Yes": int(learning),
        "School_Type_Public": 1 if school == "Public" else 0
    }])

    input_df = input_df[feature]

    # -----------------------------
    # PREDICTIONS
    # -----------------------------
    predicted_score = float(reg_model.predict(input_df)[0])
    predicted_score = max(0, min(100, predicted_score))

    result = int(pass_model.predict(input_df)[0])

    # -----------------------------
    # GRADE SYSTEM
    # -----------------------------
    if predicted_score >= 90:
        grade = "A"
    elif predicted_score >= 80:
        grade = "B"
    elif predicted_score >= 70:
        grade = "C"
    elif predicted_score >= 60:
        grade = "D"
    else:
        grade = "F"

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📊 Score", f"{predicted_score:.1f}/100")
    col2.metric("🎯 Status", "PASS ✅" if result == 1 else "FAIL ❌")
    col3.metric("🏅 Grade", grade)

    st.progress(int(predicted_score))

    # -----------------------------
    # RECOMMENDATIONS
    # -----------------------------
    st.subheader("💡 Recommendations")

    recommendations = []

    if attendance < 75:
        recommendations.append("Improve attendance consistency.")
    if hours < 10:
        recommendations.append("Increase study hours.")
    if sleep < 7:
        recommendations.append("Fix sleep schedule.")
    if tutoring == 0:
        recommendations.append("Join tutoring sessions.")
    if motivation == "Low":
        recommendations.append("Work on motivation habits.")
    if resources == "Low":
        recommendations.append("Improve learning resources.")

    if not recommendations:
        recommendations.append("Excellent performance — keep it up!")

    for r in recommendations:
        st.info(r)

    st.success("Prediction completed successfully.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer-box">
Built for educational analytics • AI-powered predictions
</div>
""", unsafe_allow_html=True)