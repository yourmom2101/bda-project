# Property Price Prediction Model

## Project Overview
This project implements a machine learning model for predicting property prices using XGBoost. The model achieves high accuracy (R² = 0.9818) and provides comprehensive documentation for exam purposes.

## Files in this Repository
- `harvard_quality_model.py`: The main model implementation
- `generate_documentation.py`: Script to generate project documentation
- `exam_documentation.md`: Generated documentation for the exam
- `README.md`: This file

## How to Use

1. **Run the Model**:
```bash
python harvard_quality_model.py
```

2. **Generate Documentation**:
```bash
python generate_documentation.py
```

3. **View Documentation**:
The documentation will be saved in `exam_documentation.md`

## Group Members
Tobias, Christian, Maria, Nikola

## Project Structure
- Data preprocessing and cleaning
- Feature engineering
- Model training and evaluation
- Documentation generation
- Visualization tools

## Requirements
- Python 3.x
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn
  - joblib

## Installation
```bash
pip install -r requirements.txt
```

## Contact
[Add your contact information here] 

## Limitations and Potential Improvements

### Limitations

1. **Data Dependency:**
   - The model relies heavily on the quality and consistency of the input data. Missing values, outliers, or errors can significantly impact performance.
   - It assumes that the features (e.g., `Fin_sqft`, `Lotsize`, `Year_Built`) are available and correctly formatted in new datasets.

2. **Feature Engineering:**
   - The current feature set is based on domain knowledge, but it may not capture all relevant relationships. For example, location-based features (e.g., proximity to amenities, schools, or public transport) are missing.
   - Some features (e.g., `Property_Age`, `Total_Bathrooms`) are derived from raw data, which may introduce noise if the raw data is inaccurate.

3. **Model Complexity:**
   - The model uses XGBoost, which is powerful but can be prone to overfitting if not tuned carefully. The current hyperparameters are set to reasonable defaults, but they may not be optimal for all datasets.
   - The model does not explicitly account for non-linear relationships or interactions between features, which may limit its performance on complex datasets.

4. **Time Series Considerations:**
   - While the model uses time-based features (e.g., `Sale_Year`, `Sale_Month`), it does not fully leverage time series techniques (e.g., ARIMA, LSTM) that could better capture temporal trends and seasonality.

5. **Interpretability:**
   - XGBoost is a "black-box" model, making it difficult to interpret how individual features influence predictions. This can be a limitation in scenarios where explainability is crucial.

6. **Generalization:**
   - The model may not generalize well to entirely new markets or regions where property dynamics differ significantly from the training data.

### Potential Improvements

1. **Enhanced Feature Engineering:**
   - **Location-Based Features:** Incorporate geographic data (e.g., distance to schools, parks, public transport) to capture location-based influences on property prices.
   - **Market Indicators:** Include broader market indicators (e.g., interest rates, economic growth) to account for macroeconomic factors.
   - **Interaction Terms:** Create interaction features (e.g., `Fin_sqft * Year_Built`) to capture non-linear relationships.

2. **Advanced Data Preprocessing:**
   - **Imputation Techniques:** Use more sophisticated imputation methods (e.g., KNN imputation) to handle missing values.
   - **Outlier Detection:** Implement advanced outlier detection techniques (e.g., Isolation Forest) to better identify and handle anomalies.

3. **Model Enhancements:**
   - **Hyperparameter Tuning:** Use techniques like Bayesian optimization or grid search to find optimal hyperparameters for XGBoost.
   - **Ensemble Methods:** Combine multiple models (e.g., XGBoost, Random Forest, LightGBM) to improve robustness and accuracy.
   - **Time Series Models:** Integrate time series models (e.g., ARIMA, LSTM) to better capture temporal trends and seasonality.

4. **Interpretability Improvements:**
   - **SHAP Values:** Use SHAP (SHapley Additive exPlanations) to provide detailed feature importance and interpretability.
   - **LIME:** Implement LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions.

5. **Cross-Validation and Validation:**
   - **Stratified Cross-Validation:** Use stratified cross-validation to ensure balanced representation of different property types or regions.
   - **External Validation:** Validate the model on entirely new datasets to assess its generalization capabilities.

6. **Deployment and Monitoring:**
   - **Model Monitoring:** Implement continuous monitoring to detect drift or degradation in model performance over time.
   - **Automated Retraining:** Set up automated retraining pipelines to update the model with new data periodically.
