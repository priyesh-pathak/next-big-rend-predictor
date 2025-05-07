import streamlit as st
import pandas as pd
from datetime import datetime
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ========== Load the trained model and prepare features ==========
@st.cache_data
def load_model_and_features():
    # Re-train the model (for simplicity, include training here)
    df = pd.read_csv('D:\trend predictor\2021_October_all_trends_data.csv')
    df = df.drop_duplicates()
    df = df.dropna(subset=['tweet_volume'])
    df['is_viral'] = (df['tweet_volume'] >= df['tweet_volume'].quantile(0.75)).astype(int)
    df['searched_at_datetime'] = pd.to_datetime(df['searched_at_datetime'])
    df['hour'] = df['searched_at_datetime'].dt.hour
    df['day_of_week'] = df['searched_at_datetime'].dt.dayofweek
    df = pd.get_dummies(df, columns=['searched_in_country'], drop_first=True)
    
    features = ['hour', 'day_of_week'] + [col for col in df.columns if col.startswith('searched_in_country_')]
    X = df[features]
    y = df['is_viral']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, features

model, features = load_model_and_features()

# ========== Streamlit UI ==========
st.title("🔮 Viral Trend Predictor")

st.markdown("Enter details about a Twitter trend to predict whether it might go viral.")

trend_name = st.text_input("Trend Name", value="ExampleTrend")
tweet_volume = st.number_input("Tweet Volume", min_value=0, value=150000)
country = st.selectbox("Country", options=["India", "USA", "United Kingdom", "Germany", "Japan", "France", "Canada"])
searched_time = datetime.now()
hour = searched_time.hour
day_of_week = searched_time.weekday()

# ========== Prepare input ==========
input_data = pd.DataFrame([{
    'hour': hour,
    'day_of_week': day_of_week,
    **{col: 0 for col in features if col.startswith('searched_in_country_')}
}])

# Match country column
country_col = f'searched_in_country_{country}'
if country_col in input_data.columns:
    input_data[country_col] = 1

# ========== Predict ==========
if st.button("Predict Trend Virality"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success(f"🔥 The trend '{trend_name}' is likely to go VIRAL!")
    else:
        st.info(f"📉 The trend '{trend_name}' may not go viral.")

