import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from preprocessing import QueueFeatureEngineer, get_preprocessing_pipeline

def generate_synthetic_campus_data(n_samples=2500):
    """
    Generates synthetic training data reflecting campus eatery dynamics:
    Operating hours (9-17), student lunch rush (11:30-14:30), and 2 PM staff break.
    """
    np.random.seed(42)
    
    hours = np.random.uniform(9.0, 17.0, n_samples)
    queue_lengths = np.random.poisson(lam=14, size=n_samples)
    locations = np.random.choice(
        ["Main Campus Cafeteria", "Student Food Court Counter 1", "Student Food Court Counter 2", "Administration & Admissions Dining"],
        size=n_samples
    )
    days = np.random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], size=n_samples)
    special_events = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    
    df = pd.DataFrame({
        'hour_of_day': hours,
        'current_queue_length': queue_lengths,
        'service_location': locations,
        'day_of_week': days,
        'special_event': special_events
    })
    
    # Calculate target wait time based on domain rules + noise
    base_wait = df['current_queue_length'] * 1.8
    
    # Rush multipliers
    lunch_rush = df['hour_of_day'].between(11.5, 14.5)
    staff_break = df['hour_of_day'].between(13.75, 14.25)
    
    multiplier = np.ones(n_samples)
    multiplier[lunch_rush] *= 1.6
    multiplier[staff_break] *= 1.8
    multiplier[df['special_event'] == 1] *= 1.25
    
    df['wait_time_minutes'] = base_wait * multiplier + np.random.normal(0, 2.0, n_samples)
    df['wait_time_minutes'] = df['wait_time_minutes'].clip(lower=1.0) # Ensure non-negative wait times
    
    return df

def train_and_save_model():
    print("Generating synthetic campus queue training data...")
    df = generate_synthetic_campus_data()
    
    X = df.drop(columns=['wait_time_minutes'])
    y = df['wait_time_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Building ML Pipeline...")
    pipeline = Pipeline(steps=[
        ('feature_engineer', QueueFeatureEngineer(hour_col='hour_of_day')),
        ('preprocessor', get_preprocessing_pipeline()),
        ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42))
    ])
    
    print("Training XGBoost Regressor...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation -> RMSE: {rmse:.2f} mins | MAE: {mae:.2f} mins | R² Score: {r2:.3f}")
    
    # Save model artifact
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "queue_model_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Pipeline successfully saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
