import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
# Load data
df = pd.read_csv('/Users/nikolazalcmane/Documents/BDA/cleaned_property_sales_data.csv')

print("Cleaning data...")
# Basic cleaning
df = df[
    (df['Fin_sqft'] > 0) & 
    (df['Sale_price'] > 0) & 
    (df['Lotsize'] > 0)
]

print("Engineering features...")
# Focus on the most important features
df['Property_Age'] = 2024 - df['Year_Built']
df['Total_Bathrooms'] = df['Fbath'] + df['Hbath']
df['Price_per_sqft'] = df['Sale_price'] / df['Fin_sqft']
df['Log_Sqft'] = np.log1p(df['Fin_sqft'])
df['Log_Lotsize'] = np.log1p(df['Lotsize'])

# Select only the most important features based on previous analysis
features = [
    'Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 
    'Price_per_sqft', 'Log_Sqft', 'Log_Lotsize'
]
target = 'Sale_price'

X = df[features]
y = df[target]

print("Splitting data...")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling features...")
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training models...")
# Initialize models with optimized parameters
models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=50,  # Reduced from 200
        learning_rate=0.1,
        max_depth=4,     # Reduced from 6
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    'Random Forest': RandomForestRegressor(
        n_estimators=50,  # Reduced from 200
        max_depth=6,     # Reduced from 10
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
}

# Train and evaluate each model
results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2')  # Reduced from 5 folds
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2,
        'CV R² Mean': cv_mean,
        'CV R² Std': cv_std
    })
    
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Cross-validation R²: {cv_mean:.4f} (±{cv_std:.4f})")

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Plot results
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='R² Score', data=results_df)
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance from XGBoost
if 'XGBoost' in models:
    xgb_model = models['XGBoost']
    importances = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(feature_importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show() 