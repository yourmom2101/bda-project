import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def format_currency(x, p):
    return f'${x:,.0f}'

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Property Prices')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model_pipeline):
    feature_names = {
        'Fin_sqft': 'Finished Square Feet',
        'Log_Sqft': 'Log of Square Feet',
        'Price_per_sqft': 'Price per Square Foot',
        'Lotsize': 'Lot Size',
        'Property_Age': 'Property Age',
        'Total_Bathrooms': 'Total Bathrooms',
        'Bdrms': 'Number of Bedrooms',
        'Stories': 'Number of Stories',
        'Bathrooms_per_Bedroom': 'Bathrooms per Bedroom',
        'Sqft_per_Bedroom': 'Square Feet per Bedroom',
        'Lot_to_Sqft_Ratio': 'Lot to Square Feet Ratio',
        'Age_Squared': 'Property Age Squared',
        'Log_Lotsize': 'Log of Lot Size',
        'Total_Rooms': 'Total Rooms',
        'Room_Density': 'Room Density',
        'Sqft_per_Story': 'Square Feet per Story',
        'Bathroom_Ratio': 'Bathroom Ratio'
    }
    
    feature_importance = pd.DataFrame({
        'Feature': model_pipeline.named_steps['feature_selector'].get_feature_names_out(),
        'Importance': model_pipeline.named_steps['model'].feature_importances_
    })
    
    feature_importance['Feature'] = feature_importance['Feature'].map(feature_names)
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), color='skyblue')
    plt.title('Top 10 Most Important Features in Predicting Property Prices')
    plt.xlabel('Relative Importance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred, residuals, alpha=0.5, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price ($)')
    plt.ylabel('Residual (Actual - Predicted) ($)')
    plt.title('Residuals Plot: Prediction Errors vs Predicted Prices')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    plt.tight_layout()
    plt.savefig('plots/residuals_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals, kde=True, color='purple')
    plt.xlabel('Residual (Actual - Predicted) ($)')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_currency))
    plt.tight_layout()
    plt.savefig('plots/residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_relationships(X_test, y_test):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.scatterplot(data=pd.DataFrame({'Fin_sqft': X_test['Fin_sqft'], 'Price': y_test}), 
                   x='Fin_sqft', y='Price', alpha=0.5, ax=axes[0, 0])
    axes[0, 0].set_title('Finished Square Feet vs Price')
    axes[0, 0].set_xlabel('Finished Square Feet')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True)
    
    sns.boxplot(data=pd.DataFrame({'Bdrms': X_test['Bdrms'], 'Price': y_test}), 
                x='Bdrms', y='Price', ax=axes[0, 1])
    axes[0, 1].set_title('Number of Bedrooms vs Price')
    axes[0, 1].set_xlabel('Number of Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].grid(True)
    
    sns.scatterplot(data=pd.DataFrame({'Property_Age': X_test['Property_Age'], 'Price': y_test}), 
                   x='Property_Age', y='Price', alpha=0.5, ax=axes[1, 0])
    axes[1, 0].set_title('Property Age vs Price')
    axes[1, 0].set_xlabel('Property Age (Years)')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].grid(True)
    
    sns.boxplot(data=pd.DataFrame({'Total_Bathrooms': X_test['Total_Bathrooms'], 'Price': y_test}), 
                x='Total_Bathrooms', y='Price', ax=axes[1, 1])
    axes[1, 1].set_title('Total Bathrooms vs Price')
    axes[1, 1].set_xlabel('Number of Bathrooms')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_plots(X_test, y_test, y_pred, model_pipeline):
    os.makedirs('plots', exist_ok=True)
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model_pipeline)
    plot_residuals(y_test, y_pred)
    plot_feature_relationships(X_test, y_test) 