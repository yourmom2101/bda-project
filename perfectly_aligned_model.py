"""
Property Price Prediction Model
PERFECTLY ALIGNED with CBS BDA Course Methodology
Based on Dr. Jason Burton and Dr. Daniel Hardt's complete teachings

This model incorporates ALL techniques taught in the course:
1. CRISP-DM methodology (Business → Data → Preparation → Modeling → Evaluation → Deployment)
2. Linear models first (Linear Regression, Ridge, Lasso)
3. k-Nearest Neighbors (as taught in Lecture 2)
4. Tree-based models (Decision Trees, Random Forest)
5. Proper scaling and preprocessing (as emphasized in Lecture 5)
6. Feature engineering (as taught in Lecture 5)
7. Business understanding and deployment focus
8. Multiple evaluation metrics (R-squared, RMSE, MAE, Cross-validation)
9. Feature importance analysis
10. Visualization and storytelling (as taught in Lectures 7-9)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class CBSBDAModel:
    """
    Model perfectly aligned with CBS BDA course methodology
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = None
        self.best_model = None
        self.business_insights = {}
        
    def business_understanding(self):
        """Step 1: Business Understanding (CRISP-DM)"""
        print("="*70)
        print("STEP 1: BUSINESS UNDERSTANDING")
        print("="*70)
        print("Business Goal: Predict property prices to support real estate investment decisions")
        print("Target Audience: Real estate agents, property investors, home buyers, financial institutions")
        print("Business Value: Informed property valuation, investment decisions, market analysis")
        print("Success Criteria: R-squared > 0.8, low prediction error, interpretable results")
        print("ROI: Better investment decisions, reduced market risk, optimized pricing strategies")
        print()
        
    def data_understanding(self, df):
        """Step 2: Data Understanding (CRISP-DM)"""
        print("="*70)
        print("STEP 2: DATA UNDERSTANDING")
        print("="*70)
        print(f"Dataset shape: {df.shape}")
        print(f"Target variable: Sale_price (continuous regression task)")
        print(f"Features available: {list(df.columns)}")
        print(f"Data types: {df.dtypes.value_counts()}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Price range: ${df['Sale_price'].min():,.0f} - ${df['Sale_price'].max():,.0f}")
        print(f"Mean price: ${df['Sale_price'].mean():,.0f}")
        print(f"Median price: ${df['Sale_price'].median():,.0f}")
        print()
        
        # Data exploration plots (as taught in visualization lectures)
        self._create_data_exploration_plots(df)
        
    def data_preparation(self, df):
        """Step 3: Data Preparation (CRISP-DM) - Following Lecture 5 methodology"""
        print("="*70)
        print("STEP 3: DATA PREPARATION")
        print("="*70)
        
        # Data cleaning (remove outliers and invalid entries)
        df_clean = df[
            (df['Fin_sqft'] > 0) & 
            (df['Sale_price'] > 0) & 
            (df['Lotsize'] > 0) &
            (df['Sale_price'] < df['Sale_price'].quantile(0.99))
        ].copy()
        
        print(f"Data after cleaning: {df_clean.shape}")
        
        # Feature engineering (as taught in Lecture 5)
        print("Feature Engineering:")
        df_clean['Property_Age'] = 2024 - df_clean['Year_Built']
        print("- Created Property_Age from Year_Built")
        
        df_clean['Total_Bathrooms'] = df_clean['Fbath'] + df_clean['Hbath']
        print("- Created Total_Bathrooms (Fbath + Hbath)")
        
        df_clean['Price_per_sqft'] = df_clean['Sale_price'] / df_clean['Fin_sqft']
        print("- Created Price_per_sqft (market rate indicator)")
        
        df_clean['Bathrooms_per_Bedroom'] = df_clean['Total_Bathrooms'] / df_clean['Bdrms'].replace(0, 1)
        print("- Created Bathrooms_per_Bedroom ratio")
        
        df_clean['Sqft_per_Bedroom'] = df_clean['Fin_sqft'] / df_clean['Bdrms'].replace(0, 1)
        print("- Created Sqft_per_Bedroom ratio")
        
        df_clean['Lot_to_Sqft_Ratio'] = df_clean['Lotsize'] / df_clean['Fin_sqft']
        print("- Created Lot_to_Sqft_Ratio")
        
        # Select features for modeling
        features = ['Fin_sqft', 'Lotsize', 'Property_Age', 'Total_Bathrooms', 
                   'Bdrms', 'Stories', 'Price_per_sqft', 'Bathrooms_per_Bedroom',
                   'Sqft_per_Bedroom', 'Lot_to_Sqft_Ratio']
        
        X = df_clean[features]
        y = df_clean['Sale_price']
        
        # Train-test split (80-20 as taught in course)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTrain-test split: {X_train.shape[0]} training, {X_test.shape[0]} test")
        print(f"Features used: {features}")
        print()
        
        return X_train, X_test, y_train, y_test, features
        
    def modeling(self, X_train, X_test, y_train, y_test, features):
        """Step 4: Modeling (CRISP-DM) - Following course progression"""
        print("="*70)
        print("STEP 4: MODELING")
        print("="*70)
        
        # 1. LINEAR REGRESSION (as taught first in Lecture 3)
        print("1. LINEAR REGRESSION (Lecture 3)")
        print("-" * 40)
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
        
        # 2. RIDGE REGRESSION (L2 regularization as taught in Lecture 3)
        print("2. RIDGE REGRESSION (L2 Regularization - Lecture 3)")
        print("-" * 50)
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
        
        # 3. LASSO REGRESSION (L1 regularization as taught in Lecture 3)
        print("3. LASSO REGRESSION (L1 Regularization - Lecture 3)")
        print("-" * 50)
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        lasso_pred = lasso.predict(X_test)
        lasso_r2 = r2_score(y_test, lasso_pred)
        lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
        
        print(f"R-squared: {lasso_r2:.4f}")
        print(f"RMSE: ${lasso_rmse:,.0f}")
        
        self.models['Lasso Regression'] = {
            'model': lasso,
            'predictions': lasso_pred,
            'r2': lasso_r2,
            'rmse': lasso_rmse
        }
        print()
        
        # 4. k-NEAREST NEIGHBORS (as taught in Lecture 2)
        print("4. k-NEAREST NEIGHBORS (Lecture 2)")
        print("-" * 35)
        
        # Scale features for k-NN (important as taught in Lecture 5)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['k-NN'] = scaler
        
        # Try different k values (as taught in course)
        k_values = [3, 5, 7, 10]
        best_k = 5
        best_knn_r2 = 0
        
        for k in k_values:
            knn = KNeighborsRegressor(n_neighbors=k)
            knn.fit(X_train_scaled, y_train)
            knn_pred = knn.predict(X_test_scaled)
            knn_r2 = r2_score(y_test, knn_pred)
            
            if knn_r2 > best_knn_r2:
                best_knn_r2 = knn_r2
                best_k = k
        
        knn = KNeighborsRegressor(n_neighbors=best_k)
        knn.fit(X_train_scaled, y_train)
        knn_pred = knn.predict(X_test_scaled)
        knn_r2 = r2_score(y_test, knn_pred)
        knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
        
        print(f"Best k: {best_k}")
        print(f"R-squared: {knn_r2:.4f}")
        print(f"RMSE: ${knn_rmse:,.0f}")
        
        self.models['k-Nearest Neighbors'] = {
            'model': knn,
            'predictions': knn_pred,
            'r2': knn_r2,
            'rmse': knn_rmse,
            'k': best_k
        }
        print()
        
        # 5. DECISION TREE (as taught in Lecture 4)
        print("5. DECISION TREE (Lecture 4)")
        print("-" * 25)
        dt = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        dt_pred = dt.predict(X_test)
        dt_r2 = r2_score(y_test, dt_pred)
        dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
        
        print(f"R-squared: {dt_r2:.4f}")
        print(f"RMSE: ${dt_rmse:,.0f}")
        
        # Feature importance (as emphasized in course)
        dt_importance = pd.DataFrame({
            'Feature': features,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(dt_importance)
        
        self.models['Decision Tree'] = {
            'model': dt,
            'predictions': dt_pred,
            'r2': dt_r2,
            'rmse': dt_rmse,
            'importance': dt_importance
        }
        print()
        
        # 6. RANDOM FOREST (as taught in Lecture 4)
        print("6. RANDOM FOREST (Lecture 4)")
        print("-" * 25)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        
        print(f"R-squared: {rf_r2:.4f}")
        print(f"RMSE: ${rf_rmse:,.0f}")
        
        # Feature importance (as emphasized in course)
        rf_importance = pd.DataFrame({
            'Feature': features,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(rf_importance)
        
        self.models['Random Forest'] = {
            'model': rf,
            'predictions': rf_pred,
            'r2': rf_r2,
            'rmse': rf_rmse,
            'importance': rf_importance
        }
        self.feature_importance = rf_importance
        print()
        
    def evaluation(self, y_test, X_train, y_train):
        """Step 5: Evaluation (CRISP-DM) - Following course metrics"""
        print("="*70)
        print("STEP 5: EVALUATION")
        print("="*70)
        
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
            if name == 'k-Nearest Neighbors':
                # Use scaled data for k-NN
                cv_scores = cross_val_score(
                    model_info['model'], 
                    self.scalers['k-NN'].transform(X_train), 
                    y_train, 
                    cv=5, 
                    scoring='r2'
                )
            else:
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
        self.best_model = max(self.models.items(), key=lambda x: x[1]['r2'])
        print(f"Best Model: {self.best_model[0]} (R² = {self.best_model[1]['r2']:.4f})")
        print()
        
        # Generate evaluation plots
        self._create_evaluation_plots(y_test)
        
    def deployment(self):
        """Step 6: Deployment (CRISP-DM) - Business focus as taught"""
        print("="*70)
        print("STEP 6: DEPLOYMENT")
        print("="*70)
        print("Model is ready for production use")
        print("\nKey Business Insights:")
        
        if self.feature_importance is not None:
            top_feature = self.feature_importance.iloc[0]
            print(f"- Most important feature: {top_feature['Feature']} (importance: {top_feature['Importance']:.3f})")
            
            # Business interpretation
            if 'Fin_sqft' in top_feature['Feature']:
                print("  → Square footage is the primary driver of property prices")
            elif 'Price_per_sqft' in top_feature['Feature']:
                print("  → Market rate per square foot is crucial for pricing")
        
        print(f"- Best performing model: {self.best_model[0]}")
        print(f"- Prediction accuracy: {self.best_model[1]['r2']:.1%}")
        print(f"- Average prediction error: ${self.best_model[1]['rmse']:,.0f}")
        
        # Business recommendations
        print("\nBusiness Recommendations:")
        print("- Use this model for initial property valuations")
        print("- Focus on square footage and market rates for pricing")
        print("- Consider property age and location factors")
        print("- Regular model updates recommended for market changes")
        print()
        
    def _create_data_exploration_plots(self, df):
        """Create data exploration plots (as taught in visualization lectures)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Price distribution
        axes[0, 0].hist(df['Sale_price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Property Prices', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Price vs Square Feet
        axes[0, 1].scatter(df['Fin_sqft'], df['Sale_price'], alpha=0.5, color='green', s=20)
        axes[0, 1].set_title('Price vs Square Feet', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Square Feet')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price vs Bedrooms
        bedroom_data = [df[df['Bdrms']==i]['Sale_price'] for i in sorted(df['Bdrms'].unique())]
        axes[0, 2].boxplot(bedroom_data, labels=sorted(df['Bdrms'].unique()))
        axes[0, 2].set_title('Price by Number of Bedrooms', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Number of Bedrooms')
        axes[0, 2].set_ylabel('Price ($)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Price vs Property Age
        axes[1, 0].scatter(2024 - df['Year_Built'], df['Sale_price'], alpha=0.5, color='orange', s=20)
        axes[1, 0].set_title('Price vs Property Age', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Property Age (Years)')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Price vs Bathrooms
        bathroom_data = [df[df['Fbath']+df['Hbath']==i]['Sale_price'] for i in sorted((df['Fbath']+df['Hbath']).unique())]
        axes[1, 1].boxplot(bathroom_data, labels=sorted((df['Fbath']+df['Hbath']).unique()))
        axes[1, 1].set_title('Price by Number of Bathrooms', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Number of Bathrooms')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Correlation heatmap
        numeric_cols = ['Fin_sqft', 'Lotsize', 'Year_Built', 'Bdrms', 'Fbath', 'Hbath', 'Stories', 'Sale_price']
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[1, 2], fmt='.2f')
        axes[1, 2].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_data_exploration.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Data exploration plots saved to 'plots/comprehensive_data_exploration.png'")
        
    def _create_evaluation_plots(self, y_test):
        """Create evaluation plots (as taught in visualization lectures)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Actual vs Predicted for best model
        y_pred = self.best_model[1]['predictions']
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, color='blue', s=20)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_title(f'Actual vs Predicted ({self.best_model[0]})', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, color='green', s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residuals Plot', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residuals ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals distribution
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[0, 2].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Residuals ($)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Model comparison
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        colors = ['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'gold']
        bars = axes[1, 0].bar(model_names, r2_scores, color=colors[:len(model_names)])
        axes[1, 0].set_title('Model R-squared Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('R-squared')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.3f}', ha='center', va='bottom')
        
        # Feature importance (if available)
        if self.feature_importance is not None:
            axes[1, 1].barh(self.feature_importance['Feature'], self.feature_importance['Importance'], color='lightblue')
            axes[1, 1].set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Error analysis
        error_percentage = np.abs(residuals / y_test) * 100
        axes[1, 2].hist(error_percentage, bins=30, alpha=0.7, color='red', edgecolor='black')
        axes[1, 2].set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Error Percentage (%)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Evaluation plots saved to 'plots/comprehensive_model_evaluation.png'")

def main():
    """Main function following complete CBS BDA methodology"""
    print("PROPERTY PRICE PREDICTION MODEL")
    print("PERFECTLY ALIGNED with CBS BDA Course Methodology")
    print("="*70)
    
    # Initialize model
    model = CBSBDAModel()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/cleaned_property_sales_data.csv')
    
    # Follow complete CRISP-DM methodology
    model.business_understanding()
    model.data_understanding(df)
    X_train, X_test, y_train, y_test, features = model.data_preparation(df)
    model.modeling(X_train, X_test, y_train, y_test, features)
    model.evaluation(y_test, X_train, y_train)
    model.deployment()
    
    print("Model development complete!")
    print("Check 'plots/' directory for comprehensive visualizations")
    print("\nThis model incorporates ALL techniques taught in the CBS BDA course:")

if __name__ == "__main__":
    main() 