import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. Define the input data structure (This stays the same) ---
# We still accept normal, human-friendly numbers from the user.
class VideoFeatures(BaseModel):
    title: str
    description: str
    category_id: int
    duration_in_minutes: int
    subscriber_count: int
    channel_video_count: int
    channel_view_count: int
    channel_age_days: int
    publish_hour: int
    publish_day_of_week: int

# --- 2. Initialize the FastAPI App ---
app = FastAPI(title="YTML Prediction API")

# --- 3. Load the Machine Learning Assets ---
model_folder = 'models'
try:
    classifier = joblib.load(os.path.join(model_folder, 'classifier.joblib'))
    regressor = joblib.load(os.path.join(model_folder, 'regressor.joblib'))
    scaler = joblib.load(os.path.join(model_folder, 'scaler.joblib'))
    print("Models and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    classifier, regressor, scaler = None, None, None

# --- 4. Feature Engineering Function (UPDATED) ---
def process_input_data(input_features: VideoFeatures):
    data = input_features.dict()
    df = pd.DataFrame([data])
    
    # --- Part A: Create calculated features ---
    df['duration_seconds'] = df['duration_in_minutes'] * 60
    df['title_length'] = df['title'].str.len()
    df['description_length'] = df['description'].str.len()
    
    # --- Part B: Apply the Log Transformation ---
    # We convert the raw counts into their log versions.
    # We use np.log1p which calculates log(1 + x) to safely handle zeros.
    df['log_subscriber_count'] = np.log1p(df['subscriber_count'])
    df['log_channel_view_count'] = np.log1p(df['channel_view_count'])
    df['log_channel_video_count'] = np.log1p(df['channel_video_count'])

    # --- Part C: Define the final feature order to match the model ---
    # This list must be in the exact same order as the one you used for training.
    final_feature_order = [
        'category_id', 'duration_seconds', 'publish_hour',
        'publish_day_of_week', 'channel_age_days', 'title_length',
        'description_length', 'log_subscriber_count',
        'log_channel_view_count', 'log_channel_video_count'
    ]
    
    # Select and reorder the columns
    df_final = df[final_feature_order]
    
    return df_final

# --- 5. Prediction Endpoint (This stays the same) ---
# It now uses the updated process_input_data function automatically.
@app.post("/predict")
def predict(input_features: VideoFeatures):
    if not all([classifier, regressor, scaler]):
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    processed_df = process_input_data(input_features)
    scaled_features = scaler.transform(processed_df)
    
    class_prediction = classifier.predict(scaled_features)[0]
    reg_prediction_log = regressor.predict(scaled_features)[0]
    
    reg_prediction = np.expm1(reg_prediction_log)
    
    class_labels = {0: 'Underperforming', 1: 'Average', 2: 'Popular', 3: 'Viral'}
    predicted_tier = class_labels.get(class_prediction, "Unknown")
    
    return {
        "predicted_engagement_tier": predicted_tier,
        "predicted_view_count": round(reg_prediction)
    }

# --- 6. Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    return {"status": "API is running"}