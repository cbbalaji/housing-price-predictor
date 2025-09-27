import pandas as pd
import joblib
import boto3
import io
import os
import s3fs

from preprocessing import load_data, new_features, loc_cluster, preprocess  # Assumes these are defined
from models import train_all_models              # Your earlier function
from sklearn.metrics import mean_squared_error

# AWS S3 config
S3_BUCKET = 'house-price-prediction-cbb'
S3_KEY = 'models/best_model_prediction.pkl'
AWS_REGION = 'us-east-2'  # Update as needed


# Data set path
file_path = "s3://house-price-prediction-cbb/housing.csv"


'''
def save_model_to_s3(model, bucket=S3_BUCKET, key=S3_KEY, region=AWS_REGION):
    # Serialize model to in-memory buffer
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    # Upload to S3
    s3 = boto3.client('s3', region_name=region)
    s3.upload_fileobj(buffer, Bucket=bucket, Key=key)
    print(f"✅ Model saved to S3: s3://{bucket}/{key}")
'''

def save_model(model, model_dir):
    import os, joblib
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'model.pkl')
    joblib.dump(model, path)
    print(f"✅ Model saved to: {path}")
    

def main(model_dir):
    print("🚀 Loading and preprocessing data...")
    df = load_data(file_path)  # Your custom function
    X_train, X_test, y_train, y_test = preprocess(df)  # Returns split data

    print("🔍 Training models...")
    results = train_all_models(X_train, y_train, X_test, y_test)

    print("\n🏆 Selecting best model...")
    best_model_name = min(results, key=lambda k: results[k]['mse'])
    best_model = results[best_model_name]['model']
    best_mse = results[best_model_name]['mse']

    print(f"✅ Best Model: {best_model_name.upper()} with MSE: {best_mse:.4f}")
    # save_model_to_s3(best_model)
    save_model(best_model, model_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()
    main(args.model_dir)