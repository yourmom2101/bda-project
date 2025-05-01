import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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

# Select features
features = ['Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 'Bdrms', 'Stories']
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

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

print("\nTraining and evaluating models...")
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
    r2 = r2_score(y_test, y_pred)
    
    # Calculate cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Store results
    results.append({
        'Model': name,
        'RMSE': rmse,
        'R² Score': r2,
        'CV R² Mean': cv_mean,
        'CV R² Std': cv_std
    })
    
    # Print coefficients
    if name == 'Linear Regression':
        coefficients = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        })
        print("\nFeature Coefficients:")
        print(coefficients.sort_values('Coefficient', ascending=False))
    
    print(f"RMSE: {rmse:.2f}")
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

# Plot cross-validation scores
plt.figure(figsize=(12, 6))
sns.barplot(x='Model', y='CV R² Mean', data=results_df)
plt.title('Cross-validation R² Scores')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show() 