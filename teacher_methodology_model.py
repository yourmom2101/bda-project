"""
Property Price Prediction Model
Following CBS BDA Course Methodology
Based on Dr. Jason Burton and Dr. Daniel Hardt's teachings

This model follows the CRISP-DM methodology and course-specific techniques:
1. Business Understanding
2. Data Understanding  
3. Data Preparation
4. Modeling (Linear models first, then trees)
5. Evaluation
6. Deployment
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class CRISPDMModel:
    """
    Model following CRISP-DM methodology as taught in CBS BDA course
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def business_understanding(self):
        """Step 1: Business Understanding"""
        print("="*60)
        print("STEP 1: BUSINESS UNDERSTANDING")
        print("="*60)
        print("Business Goal: Predict property prices to support real estate decisions")
        print("Target Audience: Real estate agents, investors, home buyers")
        print("Business Value: Informed property valuation and investment decisions")
        print("Success Criteria: High R-squared (>0.8) and low prediction error")
        print()
        
    def data_understanding(self, df):
        """Step 2: Data Understanding"""
        print("="*60)
        print("STEP 2: DATA UNDERSTANDING")
        print("="*60)
        print(f"Dataset shape: {df.shape}")
        print(f"Target variable: Sale_price")
        print(f"Features available: {list(df.columns)}")
        print(f"Data types: {df.dtypes.value_counts()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Price range: ${df['Sale_price'].min():,.0f} - ${df['Sale_price'].max():,.0f}")
        print(f"Mean price: ${df['Sale_price'].mean():,.0f}")
        print()
        
        # Basic data exploration plots
        self._plot_data_exploration(df)
        
    def data_preparation(self, df):
        """Step 3: Data Preparation"""
        print("="*60)
        print("STEP 3: DATA PREPARATION")
        print("="*60)
        
        # Clean data (remove outliers and invalid entries)
        df_clean = df[
            (df['Fin_sqft'] > 0) & 
            (df['Sale_price'] > 0) & 
            (df['Lotsize'] > 0) &
            (df['Sale_price'] < df['Sale_price'].quantile(0.99))
        ].copy()
        
        print(f"Data after cleaning: {df_clean.shape}")
        
        # Feature engineering (following course methodology)
        df_clean['Property_Age'] = 2024 - df_clean['Year_Built']
        df_clean['Total_Bathrooms'] = df_clean['Fbath'] + df_clean['Hbath']
        df_clean['Price_per_sqft'] = df_clean['Sale_price'] / df_clean['Fin_sqft']
        df_clean['Bathrooms_per_Bedroom'] = df_clean['Total_Bathrooms'] / df_clean['Bdrms'].replace(0, 1)
        
        # Select features for modeling
        features = ['Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 
                   'Bdrms', 'Stories', 'Price_per_sqft', 'Bathrooms_per_Bedroom']
        
        X = df_clean[features]
        y = df_clean['Sale_price']
        
        # Train-test split (80-20 as taught in course)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features (important for linear models)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features used: {features}")
        print()
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features
        
    def modeling(self, X_train, X_test, y_train, y_test, features):
        """Step 4: Modeling (following course progression)"""
        print("="*60)
        print("STEP 4: MODELING")
        print("="*60)
        
        # 1. LINEAR REGRESSION (as taught first in course)
        print("1. LINEAR REGRESSION")
        print("-" * 30)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        
        print(f"R-squared: {lr_r2:.4f}")
        print(f"RMSE: ${lr_rmse:,.0f}")
        print(f"Intercept: ${lr.intercept_:,.0f}")
        
        # Feature coefficients (as emphasized in course)
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient': lr.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        print("\nFeature Coefficients:")
        print(coef_df)
        
        self.models['Linear Regression'] = {
            'model': lr,
            'predictions': lr_pred,
            'r2': lr_r2,
            'rmse': lr_rmse,
            'coefficients': coef_df
        }
        print()
        
        # 2. RIDGE REGRESSION (regularization as taught)
        print("2. RIDGE REGRESSION (L2 Regularization)")
        print("-" * 40)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        ridge_r2 = r2_score(y_test, ridge_pred)
        ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
        
        print(f"R-squared: {ridge_r2:.4f}")
        print(f"RMSE: ${ridge_rmse:,.0f}")
        
        self.models['Ridge Regression'] = {
            'model': ridge,
            'predictions': ridge_pred,
            'r2': ridge_r2,
            'rmse': ridge_rmse
        }
        print()
        
        # 3. RANDOM FOREST (tree-based as taught)
        print("3. RANDOM FOREST")
        print("-" * 20)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        print(f"R-squared: {rf_r2:.4f}")
        print(f"RMSE: ${rf_rmse:,.0f}")
        
        # Feature importance (emphasized in course)
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(importance_df)
        
        self.models['Random Forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'r2': rf_r2,
            'rmse': rf_rmse,
            'importance': importance_df
        }
        self.feature_importance = importance_df
        print()
        
    def evaluation(self, y_test, X_train, y_train):
        """Step 5: Evaluation (following course metrics)"""
        print("="*60)
        print("STEP 5: EVALUATION")
        print("="*60)
        
        # Compare all models
        results = []
        for name, model_info in self.models.items():
            results.append({
                'Model': name,
                'R-squared': model_info['r2'],
                'RMSE': model_info['rmse'],
                'MAE': mean_absolute_error(y_test, model_info['predictions'])
            })
        
        results_df = pd.DataFrame(results)
        print("Model Performance Comparison:")
        print(results_df.round(4))
        print()
        
        # Cross-validation (as taught in course)
        print("Cross-Validation Results (5-fold):")
        print("-" * 40)
        for name, model_info in self.models.items():
            cv_scores = cross_val_score(
                model_info['model'], 
                X_train, 
                y_train, 
                cv=5, 
                scoring='r2'
            )
            print(f"{name}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print()
        
        # Best model selection
        best_model = max(self.models.items(), key=lambda x: x[1]['r2'])
        print(f"Best Model: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
        print()
        
        # Generate evaluation plots
        self._plot_evaluation(y_test)
        
    def deployment(self):
        """Step 6: Deployment"""
        print("="*60)
        print("STEP 6: DEPLOYMENT")
        print("="*60)
        print("Model is ready for production use")
        print("Key insights for business:")
        
        if self.feature_importance is not None:
            top_feature = self.feature_importance.iloc[0]
            print(f"- Most important feature: {top_feature['Feature']} (importance: {top_feature['Importance']:.3f})")
        
        best_model = max(self.models.items(), key=lambda x: x[1]['r2'])
        print(f"- Best performing model: {best_model[0]}")
        print(f"- Prediction accuracy: {best_model[1]['r2']:.1%}")
        print()
        
    def _plot_data_exploration(self, df):
        """Create data exploration plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price distribution
        axes[0, 0].hist(df['Sale_price'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Property Prices')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Price vs Square Feet
        axes[0, 1].scatter(df['Fin_sqft'], df['Sale_price'], alpha=0.5, color='green')
        axes[0, 1].set_title('Price vs Square Feet')
        axes[0, 1].set_xlabel('Square Feet')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Price vs Bedrooms
        axes[1, 0].boxplot([df[df['Bdrms']==i]['Sale_price'] for i in sorted(df['Bdrms'].unique())])
        axes[1, 0].set_title('Price by Number of Bedrooms')
        axes[1, 0].set_xlabel('Number of Bedrooms')
        axes[1, 0].set_ylabel('Price ($)')
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('plots/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Data exploration plots saved to 'plots/data_exploration.png'")
        
    def _plot_evaluation(self, y_test):
        """Create evaluation plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual vs Predicted for best model
        best_model = max(self.models.items(), key=lambda x: x[1]['r2'])
        y_pred = best_model[1]['predictions']
        
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_title(f'Actual vs Predicted ({best_model[0]})')
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        
        # Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        axes[1, 0].bar(model_names, r2_scores, color=['skyblue', 'lightgreen', 'orange'])
        axes[1, 0].set_title('Model R-squared Comparison')
        axes[1, 0].set_ylabel('R-squared')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Feature importance (if available)
        if self.feature_importance is not None:
            axes[1, 1].barh(self.feature_importance['Feature'], self.feature_importance['Importance'])
            axes[1, 1].set_title('Feature Importance (Random Forest)')
            axes[1, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig('plots/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Evaluation plots saved to 'plots/model_evaluation.png'")

def main():
    """Main function following CRISP-DM methodology"""
    print("PROPERTY PRICE PREDICTION MODEL")
    print("Following CBS BDA Course Methodology")
    print("="*60)
    
    # Initialize model
    model = CRISPDMModel()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/cleaned_property_sales_data.csv')
    
    # Follow CRISP-DM steps
    model.business_understanding()
    model.data_understanding(df)
    X_train, X_test, y_train, y_test, features = model.data_preparation(df)
    model.modeling(X_train, X_test, y_train, y_test, features)
    model.evaluation(y_test, X_train, y_train)
    model.deployment()
    
    print("Model development complete!")
    print("Check 'plots/' directory for visualizations")

if __name__ == "__main__":
    main() 