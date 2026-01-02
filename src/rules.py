import joblib
import pandas as pd

def load_encoders(model_dir='models'):
    le_request_type = joblib.load(f'{model_dir}/le_request_type.pkl')
    le_urgency = joblib.load(f'{model_dir}/le_urgency.pkl')
    le_severity = joblib.load(f'{model_dir}/le_severity.pkl')
    le_location = joblib.load(f'{model_dir}/le_location.pkl')
    le_time = joblib.load(f'{model_dir}/le_time.pkl')
    le_weather = joblib.load(f'{model_dir}/le_weather.pkl')
    le_priority = joblib.load(f'{model_dir}/le_priority.pkl')
    return le_request_type, le_urgency, le_severity, le_location, le_time, le_weather, le_priority

def apply_rules(ml_priority, features, explanations):
    final_priority = ml_priority
    reasons = []

    # Rule 1: If urgency is Urgent, priority High
    if features['urgency_level'] == 'Urgent':
        final_priority = 'High'
        reasons.append("Urgency level is Urgent, setting priority to High.")

    # Rule 2: If severity is Severe, priority High
    if features['severity_level'] == 'Severe':
        final_priority = 'High'
        reasons.append("Severity level is Severe, setting priority to High.")

    # Rule 3: If impact_scope > 100, increase priority
    if features['impact_scope'] > 100:
        if final_priority == 'Low':
            final_priority = 'Moderate'
        elif final_priority == 'Moderate':
            final_priority = 'High'
        reasons.append("Impact scope > 100, increasing priority.")

    # Rule 4: If location is Highway and time is Night, High priority
    if features['location_type'] == 'Highway' and features['time_reported'] == 'Night':
        final_priority = 'High'
        reasons.append("Highway issue at night, setting to High priority.")

    # Rule 5: If weather is Storm and request_type is Road Damage, High
    if features['weather_condition'] == 'Storm' and features['request_type'] == 'Road Damage':
        final_priority = 'High'
        reasons.append("Road damage during storm, setting to High priority.")

    return final_priority, reasons

def get_ml_prediction(features, model_dir='models'):
    model = joblib.load(f'{model_dir}/priority_model.pkl')
    le_req, le_urg, le_sev, le_loc, le_tim, le_wea, le_pri = load_encoders(model_dir)

    # Encode features
    encoded = {
        'request_type_encoded': le_req.transform([features['request_type']])[0],
        'urgency_encoded': le_urg.transform([features['urgency_level']])[0],
        'severity_encoded': le_sev.transform([features['severity_level']])[0],
        'impact_scope': features['impact_scope'],
        'location_encoded': le_loc.transform([features['location_type']])[0],
        'time_encoded': le_tim.transform([features['time_reported']])[0],
        'weather_encoded': le_wea.transform([features['weather_condition']])[0],
        'past_similar_reports': features['past_similar_reports']
    }

    df_encoded = pd.DataFrame([encoded])
    prediction = model.predict(df_encoded)[0]
    ml_priority = le_pri.inverse_transform([prediction])[0]
    return ml_priority

def generate_recommendations(final_priority, features):
    """
    Generate actionable recommendations based on priority and request characteristics.
    """
    recommendations = []
    
    # Priority-based recommendations
    if final_priority == 'High':
        recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Dispatch response team within 1-2 hours")
        recommendations.append("📞 Notify relevant departments and emergency services if needed")
        recommendations.append("📋 Assign dedicated team to monitor and resolve this issue")
        recommendations.append("⏰ Set up follow-up check within 24 hours to ensure resolution")
    elif final_priority == 'Moderate':
        recommendations.append("⏱️ Schedule response within 4-6 hours during business hours")
        recommendations.append("📝 Add to active queue and assign appropriate resources")
        recommendations.append("📊 Monitor progress and update status regularly")
    else:  # Low priority
        recommendations.append("📅 Schedule for next available time slot (within 24-48 hours)")
        recommendations.append("📋 Add to standard service queue")
        recommendations.append("💡 Consider batching with similar low-priority requests for efficiency")
    
    # Request type specific recommendations
    request_type = features['request_type']
    if request_type == 'Garbage Collection':
        recommendations.append("🗑️ Coordinate with waste management team")
        if features['impact_scope'] > 100:
            recommendations.append("🚛 Arrange for additional collection vehicles if needed")
    elif request_type == 'Streetlight Repair':
        recommendations.append("💡 Contact electrical maintenance team")
        if features['location_type'] == 'Highway' or features['time_reported'] == 'Night':
            recommendations.append("⚠️ Prioritize safety - ensure temporary lighting if repair delayed")
    elif request_type == 'Tree Obstruction':
        recommendations.append("🌳 Contact arborist or tree removal service")
        if features['severity_level'] == 'Severe':
            recommendations.append("🚧 Set up safety barriers around affected area")
    elif request_type == 'Road Damage':
        recommendations.append("🛣️ Coordinate with public works department")
        if features['weather_condition'] == 'Storm' or features['weather_condition'] == 'Rainy':
            recommendations.append("☔ Monitor weather conditions - may need to postpone if unsafe")
        if features['location_type'] == 'Highway':
            recommendations.append("🚦 Consider traffic management and detour planning")
    elif request_type == 'Water Leakage':
        recommendations.append("💧 Contact water utility department immediately")
        if features['severity_level'] == 'Severe':
            recommendations.append("🔧 May require shutting off water supply - notify affected residents")
    
    # Location-specific recommendations
    location = features['location_type']
    if location == 'School Zone':
        recommendations.append("🏫 Coordinate with school administration for access and timing")
        if final_priority == 'High':
            recommendations.append("👨‍👩‍👧‍👦 Ensure student safety - consider temporary closure if needed")
    elif location == 'Public Facility':
        recommendations.append("🏛️ Notify facility management and coordinate access")
    elif location == 'Highway':
        recommendations.append("🛣️ Coordinate with traffic management for lane closures if needed")
        recommendations.append("⚠️ Ensure worker safety with proper signage and barriers")
    
    # Weather-specific recommendations
    if features['weather_condition'] == 'Storm':
        recommendations.append("⛈️ Assess weather conditions before dispatch - safety first")
        recommendations.append("📱 Keep communication lines open for weather updates")
    elif features['weather_condition'] == 'Rainy':
        recommendations.append("🌧️ Ensure appropriate equipment for wet conditions")
    
    # Past similar reports recommendation
    if features['past_similar_reports'] > 5:
        recommendations.append("📈 High frequency of similar reports - consider preventive measures")
        recommendations.append("🔍 Investigate root cause to prevent recurrence")
    
    return recommendations

def hybrid_prediction(features):
    ml_priority = get_ml_prediction(features)
    final_priority, reasons = apply_rules(ml_priority, features, [])
    recommendations = generate_recommendations(final_priority, features)
    return ml_priority, final_priority, reasons, recommendations

# Example usage
if __name__ == "__main__":
    sample_features = {
        'request_type': 'Garbage Collection',
        'urgency_level': 'Urgent',
        'severity_level': 'Severe',
        'impact_scope': 130,
        'location_type': 'Public Facility',
        'time_reported': 'Afternoon',
        'weather_condition': 'Normal',
        'past_similar_reports': 7
    }
    ml_pred, final_pred, reasons, recommendations = hybrid_prediction(sample_features)
    print(f"ML Prediction: {ml_pred}")
    print(f"Final Priority: {final_pred}")
    print("Reasons:", reasons)
    print("Recommendations:", recommendations)