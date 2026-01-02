import streamlit as st
from rules import hybrid_prediction

st.title("Community Service Request and Response Management System")

st.header("Enter Request Details")

request_type = st.selectbox("Request Type", ["Garbage Collection", "Streetlight Repair", "Tree Obstruction", "Road Damage", "Water Leakage"])
urgency_level = st.selectbox("Urgency Level", ["Low", "Normal", "Urgent"])
severity_level = st.selectbox("Severity Level", ["Minor", "Moderate", "Severe"])
impact_scope = st.number_input("Impact Scope", min_value=1, max_value=200, value=50)
location_type = st.selectbox("Location Type", ["School Zone", "Residential Area", "Public Facility", "Commercial Area", "Highway"])
time_reported = st.selectbox("Time Reported", ["Morning", "Afternoon", "Night"])
weather_condition = st.selectbox("Weather Condition", ["Normal", "Rainy", "Storm"])
past_similar_reports = st.number_input("Past Similar Reports", min_value=0, max_value=10, value=0)

features = {
    'request_type': request_type,
    'urgency_level': urgency_level,
    'severity_level': severity_level,
    'impact_scope': impact_scope,
    'location_type': location_type,
    'time_reported': time_reported,
    'weather_condition': weather_condition,
    'past_similar_reports': past_similar_reports
}

if st.button("Predict Priority"):
    ml_pred, final_pred, reasons = hybrid_prediction(features)
    st.subheader("ML Prediction")
    st.write(ml_pred)
    st.subheader("Final Priority (Hybrid)")
    st.write(final_pred)
    st.subheader("Explanations")
    for reason in reasons:
        st.write(f"- {reason}")