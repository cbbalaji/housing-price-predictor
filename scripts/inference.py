import os
import joblib
import pandas as pd
import json

# Define default values for each feature
DEFAULT_FEATURES = {
    "housing_median_age": 30.0,
    "median_income": 4.5,
    "population_per_household": 2.8,
    "rooms_per_household": 5.0,
    "bedrooms_per_room": 0.2,
    "loc_cluster": 3
}

def model_fn(model_dir):
    """Load the trained model from SageMaker's model directory"""
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    """Custom input parser for structured JSON"""
    if request_content_type == 'application/json':
        data = json.loads(request_body)

        # If it's a single dict, wrap it in a list
        if isinstance(data, dict):
            data = [data]

        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def flatten_input(input_dict):
    """Flatten nested dicts like {'feature': {'value': 28}} to {'feature': 28}"""
    flat = {}
    for key, val in input_dict.items():
        if isinstance(val, dict) and "value" in val:
            flat[key] = val["value"]
        else:
            flat[key] = val
    return flat

def preprocess_input(input_data):
    """Convert input to DataFrame and apply default values"""
    if isinstance(input_data, list):
        input_data = [flatten_input(item) for item in input_data]
    else:
        input_data = [flatten_input(input_data)]

    df = pd.DataFrame(input_data)

    # Fill missing features with defaults
    for feature, default in DEFAULT_FEATURES.items():
        if feature not in df.columns:
            df[feature] = default

    # Reorder columns to match training schema
    df = df[[feature for feature in DEFAULT_FEATURES.keys()]]

    return df

def predict_fn(input_data, model):
    """Run prediction using the loaded model"""
    try:
        df = preprocess_input(input_data)
        predictions = model.predict(df)
        return predictions.tolist()
    except Exception as e:
        print("Prediction error:", str(e))
        raise e