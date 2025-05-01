import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
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

# Simple feature engineering
df['Property_Age'] = 2024 - df['Year_Built']
df['Total_Bathrooms'] = df['Fbath'] + df['Hbath']

# Select only the most important features
features = ['Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms']
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

print("Training model...")
# Initialize and train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=50,  # Very small number for quick training
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Evaluating model...")
# Evaluate model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

print("\nFeature Importances:")
# Feature importance
importances = model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
