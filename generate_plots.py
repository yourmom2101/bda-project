import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
import joblib
import os
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("Loading saved model and components...")
# Load the components
model = joblib.load('models/property_price_model.joblib')
preprocessor = joblib.load('models/preprocessor.joblib')
feature_selector = joblib.load('models/feature_selector.joblib')
selected_features = joblib.load('models/selected_features.joblib')

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load and prepare data
print("Loading and preparing test data...")
df = pd.read_csv('data/cleaned_property_sales_data.csv')

# Create derived features
df['Property_Age'] = 2024 - df['Year_Built']
df['Total_Bathrooms'] = df['Fbath'] + df['Hbath']
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

# Clean data: replace inf/-inf with NaN, then drop NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Select features and transform
# Use the same numeric features as in the original model
numeric_features = [
    'Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 'Bdrms', 'Stories',
    'Price_per_sqft', 'Bathrooms_per_Bedroom', 'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio',
    'Age_Squared', 'Log_Sqft', 'Log_Lotsize', 'Total_Rooms', 'Room_Density',
    'Sqft_per_Story', 'Bathroom_Ratio'
]
X = df[numeric_features]
X_scaled = preprocessor.transform(X)
X_selected = feature_selector.transform(X_scaled)

# Use only 50 samples for faster processing
sample_size = 50
X_sample = X_selected[:sample_size]

print("Generating SHAP explanations...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Create SHAP summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_sample, feature_names=numeric_features, plot_type="bar", show=False)
plt.title("Feature Importance (SHAP Values)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('plots/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# Create SHAP dependence plot for most important feature
top_feature_idx = np.argsort(np.abs(shap_values).mean(0))[-1]
feature_name = numeric_features[top_feature_idx]
plt.figure(figsize=(10, 6))
shap.dependence_plot(top_feature_idx, shap_values, X_sample, feature_names=numeric_features, show=False)
plt.title(f"SHAP Dependence Plot for {feature_name}", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(f'plots/shap_dependence_{feature_name}.png', dpi=300, bbox_inches='tight')
plt.close()

print("Generating LIME explanation...")
explainer = lime_tabular.LimeTabularExplainer(
    X_selected,
    feature_names=numeric_features,
    class_names=['Price'],
    mode='regression'
)

# Generate explanation for one example
exp = explainer.explain_instance(
    X_sample[0],
    model.predict,
    num_features=10
)

plt.figure(figsize=(10, 6))
exp.as_pyplot_figure()
plt.title("LIME Explanation for Example Prediction", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig('plots/lime_explanation.png', dpi=300, bbox_inches='tight')
plt.close()

# Create comprehensive visualizations
print("\nGenerating comprehensive visualizations...")
plt.style.use('default')  # Use default matplotlib style
fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# Get predictions for the sample
y_pred = model.predict(X_sample)

# 1. Actual vs Predicted Values
axes[0, 0].scatter(df['Sale_price'][:sample_size], y_pred, alpha=0.5, color='blue')
axes[0, 0].plot([df['Sale_price'].min(), df['Sale_price'].max()], 
                [df['Sale_price'].min(), df['Sale_price'].max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Values')
axes[0, 0].set_ylabel('Predicted Values')
axes[0, 0].set_title('Actual vs Predicted Values')
axes[0, 0].grid(True)

# 2. Feature Importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Top 10 Feature Importances')
axes[0, 1].grid(True)

# 3. Residuals Plot
residuals = df['Sale_price'][:sample_size] - y_pred
axes[1, 0].scatter(y_pred, residuals, alpha=0.5, color='green')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Values')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Residuals Plot')
axes[1, 0].grid(True)

# 4. Residuals Distribution
sns.histplot(residuals, kde=True, ax=axes[1, 1], color='purple')
axes[1, 1].set_title('Residuals Distribution')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('plots/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nGenerating additional visualizations...")

# 1. Correlation Heatmap
plt.figure(figsize=(15, 12))
correlation_matrix = df[numeric_features + ['Sale_price']].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True)
plt.title('Feature Correlation Heatmap', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Price Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Original price distribution
sns.histplot(data=df, x='Sale_price', bins=50, ax=ax1)
ax1.set_title('Property Price Distribution')
ax1.set_xlabel('Price ($)')
ax1.set_ylabel('Count')

# Log-transformed price distribution
sns.histplot(data=df, x=np.log1p(df['Sale_price']), bins=50, ax=ax2)
ax2.set_title('Log-Transformed Property Price Distribution')
ax2.set_xlabel('Log(Price)')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('plots/price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Feature vs Price Scatter Plots
# Select top 4 most important features
top_features = feature_importance.head(4)['Feature'].tolist()
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    sns.scatterplot(data=df, x=feature, y='Sale_price', alpha=0.5, ax=axes[idx])
    # Add trend line
    sns.regplot(data=df, x=feature, y='Sale_price', scatter=False, color='red', ax=axes[idx])
    axes[idx].set_title(f'{feature} vs Price')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Price ($)')

plt.tight_layout()
plt.savefig('plots/feature_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Error Analysis
plt.figure(figsize=(12, 8))
# Calculate predictions for all data
y_pred_full = model.predict(X_selected)
residuals = df['Sale_price'] - y_pred_full
abs_residuals = np.abs(residuals)

# Create scatter plot with error bands
plt.scatter(y_pred_full, df['Sale_price'], alpha=0.5, c=abs_residuals, cmap='viridis')
plt.colorbar(label='Absolute Error ($)')

# Add perfect prediction line
min_val = min(df['Sale_price'].min(), y_pred_full.min())
max_val = max(df['Sale_price'].max(), y_pred_full.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

# Add error bands
std_error = np.std(residuals)
plt.fill_between([min_val, max_val], 
                 [min_val - 2*std_error, max_val - 2*std_error],
                 [min_val + 2*std_error, max_val + 2*std_error],
                 alpha=0.2, color='gray', label='±2 Standard Deviations')

plt.title('Error Analysis: Actual vs Predicted Prices')
plt.xlabel('Predicted Price ($)')
plt.ylabel('Actual Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('plots/error_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional specific scatter plots
print("\nGenerating specific feature scatter plots...")

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# 1. Stories vs Price
sns.scatterplot(data=df, x='Stories', y='Sale_price', alpha=0.5, ax=axes[0])
sns.regplot(data=df, x='Stories', y='Sale_price', scatter=False, color='red', ax=axes[0])
axes[0].set_title('Number of Stories vs Price')
axes[0].set_xlabel('Number of Stories')
axes[0].set_ylabel('Price ($)')
axes[0].grid(True)

# 2. Lot Size vs Price
sns.scatterplot(data=df, x='Lotsize', y='Sale_price', alpha=0.5, ax=axes[1])
sns.regplot(data=df, x='Lotsize', y='Sale_price', scatter=False, color='red', ax=axes[1])
axes[1].set_title('Lot Size vs Price')
axes[1].set_xlabel('Lot Size (sq ft)')
axes[1].set_ylabel('Price ($)')
axes[1].grid(True)

# 3. Finished Square Feet vs Price
sns.scatterplot(data=df, x='Fin_sqft', y='Sale_price', alpha=0.5, ax=axes[2])
sns.regplot(data=df, x='Fin_sqft', y='Sale_price', scatter=False, color='red', ax=axes[2])
axes[2].set_title('Finished Square Feet vs Price')
axes[2].set_xlabel('Finished Square Feet')
axes[2].set_ylabel('Price ($)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('plots/specific_feature_scatter_plots.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Model Performance Metrics Summary
metrics = {
    'R² Score': r2_score(df['Sale_price'], y_pred_full),
    'RMSE': np.sqrt(mean_squared_error(df['Sale_price'], y_pred_full)),
    'MAE': mean_absolute_error(df['Sale_price'], y_pred_full),
    'Mean Absolute Percentage Error': np.mean(np.abs((df['Sale_price'] - y_pred_full) / df['Sale_price'])) * 100
}

# Create a text file with metrics
with open('plots/model_metrics.txt', 'w') as f:
    f.write("Model Performance Metrics:\n")
    f.write("========================\n\n")
    for metric, value in metrics.items():
        f.write(f"{metric}: {value:.4f}\n")

print("\nAll plots and metrics have been generated in the 'plots' directory!") 