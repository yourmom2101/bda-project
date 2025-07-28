import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

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

def prepare_data(df):
    feature_engineering_pipeline = Pipeline([
        ('feature_engineering', FunctionTransformer(engineer_features))
    ])
    
    df_engineered = feature_engineering_pipeline.fit_transform(df)
    X = df_engineered[get_numeric_features()]
    y = df_engineered['Sale_price']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_metrics(y_test, y_pred):
    return {
        'R² Score': r2_score(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

def save_model(model_pipeline):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, 'models/property_price_model.joblib')

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

def train_and_evaluate():
    df = load_data('data/cleaned_property_sales_data.csv')
    df = clean_data(df)
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    model_pipeline = create_pipeline()
    
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)
    
    metrics = calculate_metrics(y_test, y_pred)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    save_model(model_pipeline)
    generate_plots(X_test, y_test, y_pred, model_pipeline)

if __name__ == "__main__":
    train_and_evaluate() 