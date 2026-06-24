# src/train.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import yaml
import os

def run_training_pipeline():
    print(" Initializing Machine Learning Training Pipeline...")
    

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    test_weeks = config['pipeline']['test_size_weeks']
    target_col = config['columns']['target']
    
   
    from data_loader import SalesDataLoader
    from features import FeatureEngineer
    
    loader = SalesDataLoader()
    raw_data = loader.load_clean_resample()
    
    fe = FeatureEngineer(target_col=target_col)
    df_features = fe.build_features(raw_data)
    
  
    df_features = df_features.sort_values(by='Order Date').reset_index(drop=True)
    unique_dates = sorted(df_features['Order Date'].unique())
    
    
    split_date = unique_dates[-test_weeks]
    print(f" Splitting timeline. Training up to: {split_date.strftime('%Y-%m-%d')}. Testing final {test_weeks} weeks.")
    
    train_mask = df_features['Order Date'] < split_date
    test_mask = df_features['Order Date'] >= split_date
    
    df_train = df_features[train_mask]
    df_test = df_features[test_mask]
    

    drop_cols = ['Order Date', target_col, 'Profit', 'Discount']
    
    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=drop_cols)
    y_test = df_test[target_col]
    
    
    print(" Training the XGBoost model sequential decision trees...")
    model = XGBRegressor(
        n_estimators=100,     
        learning_rate=0.05,   
        max_depth=5,          
        n_jobs=-1           
    )
    
    model.fit(X_train, y_train)
    print(" Model training complete.")
    
    predictions = model.predict(X_test)
    
    predictions = np.clip(predictions, 0, None)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    
    print("\n ---FINAL MODEL EVALUATION METRICS ---")
    print(f" Mean Absolute Error (MAE): ${mae:.2f}")
    print(f" Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print("-----------------------------------------")
    
    
    df_test = df_test.copy()
    df_test['Actual_Sales'] = y_test
    df_test['Predicted_Sales'] = predictions
    
if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as e:
        print(f"\n Training pipeline failed. Error breakdown:\n{e}")