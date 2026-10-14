import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class QueueFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer campus-specific time features:
    - Operating hours constraint (9 AM - 5 PM)
    - Student lunch rush (11:30 AM - 2:30 PM)
    - Admissions staff break wave (~2:00 PM / 14:00)
    """
    def __init__(self, hour_col='hour_of_day'):
        self.hour_col = hour_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        
        # Binary flag for operating hours (9 AM to 5 PM)
        X_['is_operating_hours'] = X_[self.hour_col].between(9, 17).astype(int)
        
        # Binary flag for Student Lunch Rush (11:30 AM to 2:30 PM -> 11.5 to 14.5)
        X_['is_lunch_rush'] = X_[self.hour_col].between(11.5, 14.5).astype(int)
        
        # Binary flag for Admissions Staff Lunch Wave (~2:00 PM / 14:00 surge)
        X_['is_staff_break_wave'] = X_[self.hour_col].between(13.75, 14.25).astype(int)
        
        return X_

def get_preprocessing_pipeline():
    """
    Constructs a full Scikit-Learn ColumnTransformer pipeline 
    for numerical scaling and categorical encoding.
    """
    numeric_features = ['current_queue_length', 'hour_of_day']
    categorical_features = ['service_location', 'day_of_week']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor
