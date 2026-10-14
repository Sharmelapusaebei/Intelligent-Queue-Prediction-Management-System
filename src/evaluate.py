import pandas as pd
import numpy as np
import shap
import joblib
from train import generate_synthetic_campus_data

def evaluate_model_explainability():
    print("Loading test data and trained pipeline for SHAP analysis...")
    df = generate_synthetic_campus_data(n_samples=200)
    X = df.drop(columns=['wait_time_minutes'])
    
    try:
        pipeline = joblib.load("models/queue_model_pipeline.pkl")
    except FileNotFoundError:
        print("Trained model not found. Please run train.py first.")
        return

    # Extract transformed features for SHAP
    feature_engineer = pipeline.named_steps['feature_engineer']
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    X_engineered = feature_engineer.transform(X)
    X_transformed = preprocessor.transform(X_engineered)
    
    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    print("SHAP explainability engine initialized successfully.")
    return shap_values

if __name__ == "__main__":
    evaluate_model_explainability()
