import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import cross_val_score
import joblib
import os
from data_processor import create_preprocessor, get_numeric_features

def create_xgboost_model():
    return xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )

def create_pipeline():
    preprocessor = create_preprocessor()
    model = create_xgboost_model()
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', SelectFromModel(model, threshold='median')),
        ('model', model)
    ])

def calculate_metrics(y_test, y_pred):
    return {
        'R² Score': r2_score(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred),
        'RMSE': mean_squared_error(y_test, y_pred, squared=False),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': (abs((y_test - y_pred) / y_test) * 100).mean()
    }

def save_model(model_pipeline):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/property_price_model.joblib')

def evaluate_model(model_pipeline, X_train, y_train, X_test, y_test):
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return y_pred, metrics 