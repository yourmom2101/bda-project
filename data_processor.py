import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    df = pd.read_csv(file_path)
    required_columns = ['Fin_sqft', 'Sale_price', 'Lotsize', 'Year_Built', 'Fbath', 'Hbath', 'Bdrms', 'Stories']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    return df

def clean_data(df):
    return df[
        (df['Fin_sqft'] > 0) & 
        (df['Sale_price'] > 0) & 
        (df['Lotsize'] > 0) &
        (df['Sale_price'] < df['Sale_price'].quantile(0.99))
    ]

def add_basic_features(df):
    current_year = datetime.now().year
    df['Property_Age'] = current_year - df['Year_Built']
    df['Total_Bathrooms'] = df['Fbath'] + df['Hbath']
    return df

def add_derived_features(df):
    df['Price_per_sqft'] = df['Sale_price'] / df['Fin_sqft']
    df['Log_Sqft'] = np.log1p(df['Fin_sqft'])
    df['Log_Lotsize'] = np.log1p(df['Lotsize'])
    df['Bathrooms_per_Bedroom'] = df['Total_Bathrooms'] / df['Bdrms'].replace(0, 1)
    df['Sqft_per_Bedroom'] = df['Fin_sqft'] / df['Bdrms'].replace(0, 1)
    df['Lot_to_Sqft_Ratio'] = df['Lotsize'] / df['Fin_sqft']
    df['Age_Squared'] = df['Property_Age'] ** 2
    df['Total_Rooms'] = df['Bdrms'] + df['Total_Bathrooms']
    df['Room_Density'] = df['Total_Rooms'] / df['Fin_sqft']
    df['Sqft_per_Story'] = df['Fin_sqft'] / df['Stories'].replace(0, 1)
    df['Bathroom_Ratio'] = df['Total_Bathrooms'] / df['Total_Rooms'].replace(0, 1)
    return df

def engineer_features(df):
    return add_derived_features(add_basic_features(df))

def get_numeric_features():
    return [
        'Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
        'Price_per_sqft', 'Bathrooms_per_Bedroom', 'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio',
        'Age_Squared', 'Log_Sqft', 'Log_Lotsize', 'Total_Rooms', 'Room_Density',
        'Sqft_per_Story', 'Bathroom_Ratio'
    ]

def create_preprocessor():
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    return ColumnTransformer([
        ('num', numeric_transformer, get_numeric_features())
    ])

def prepare_data(df):
    feature_engineering_pipeline = Pipeline([
        ('feature_engineering', FunctionTransformer(engineer_features))
    ])
    
    df_engineered = feature_engineering_pipeline.fit_transform(df)
    X = df_engineered[get_numeric_features()]
    y = df_engineered['Sale_price']
    
    return train_test_split(X, y, test_size=0.2, random_state=42) 