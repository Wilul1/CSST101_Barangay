import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Encode categorical features
    le_request_type = LabelEncoder()
    le_urgency = LabelEncoder()
    le_severity = LabelEncoder()
    le_location = LabelEncoder()
    le_time = LabelEncoder()
    le_weather = LabelEncoder()
    le_priority = LabelEncoder()

    df['request_type_encoded'] = le_request_type.fit_transform(df['request_type'])
    df['urgency_encoded'] = le_urgency.fit_transform(df['urgency_level'])
    df['severity_encoded'] = le_severity.fit_transform(df['severity_level'])
    df['location_encoded'] = le_location.fit_transform(df['location_type'])
    df['time_encoded'] = le_time.fit_transform(df['time_reported'])
    df['weather_encoded'] = le_weather.fit_transform(df['weather_condition'])
    df['priority_encoded'] = le_priority.fit_transform(df['ml_priority'])

    # Features: request_type, urgency, severity, impact_scope, location, time, weather, past_similar_reports
    X = df[['request_type_encoded', 'urgency_encoded', 'severity_encoded', 'impact_scope', 'location_encoded', 'time_encoded', 'weather_encoded', 'past_similar_reports']]
    y = df['priority_encoded']

    return X, y, le_request_type, le_urgency, le_severity, le_location, le_time, le_weather, le_priority

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model

def save_model(model, le_request_type, le_urgency, le_severity, le_location, le_time, le_weather, le_priority, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'priority_model.pkl'))
    joblib.dump(le_request_type, os.path.join(model_dir, 'le_request_type.pkl'))
    joblib.dump(le_urgency, os.path.join(model_dir, 'le_urgency.pkl'))
    joblib.dump(le_severity, os.path.join(model_dir, 'le_severity.pkl'))
    joblib.dump(le_location, os.path.join(model_dir, 'le_location.pkl'))
    joblib.dump(le_time, os.path.join(model_dir, 'le_time.pkl'))
    joblib.dump(le_weather, os.path.join(model_dir, 'le_weather.pkl'))
    joblib.dump(le_priority, os.path.join(model_dir, 'le_priority.pkl'))

if __name__ == "__main__":
    df = load_data('data/Brgy_Bagumbayan_Synthetic_Community_Service_Requests_500.csv')
    X, y, le_req, le_urg, le_sev, le_loc, le_tim, le_wea, le_pri = preprocess_data(df)
    model = train_model(X, y)
    save_model(model, le_req, le_urg, le_sev, le_loc, le_tim, le_wea, le_pri)