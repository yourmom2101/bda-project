import numpy as np
from data_processor import load_data, clean_data, prepare_data
from model_builder import create_pipeline, evaluate_model, save_model
from visualizer import generate_plots

np.random.seed(42)

def run_pipeline():
    df = load_data('data/cleaned_property_sales_data.csv')
    df = clean_data(df)
    
    X_train, X_test, y_train, y_test = prepare_data(df)
    model_pipeline = create_pipeline()
    
    y_pred, metrics = evaluate_model(model_pipeline, X_train, y_train, X_test, y_test)
    
    save_model(model_pipeline)
    generate_plots(X_test, y_test, y_pred, model_pipeline)

if __name__ == "__main__":
    run_pipeline() 