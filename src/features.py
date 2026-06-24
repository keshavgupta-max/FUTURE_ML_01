# src/features.py
import pandas as pd
import numpy as np
import os

class FeatureEngineer:
    def __init__(self, target_col: str = "Sales"):
        self.target_col = target_col

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates time lags, rolling trends, extracts calendar variables,
        and applies structural one-hot encoding across business groupings.
        """

        df = df.sort_values(by=['Category', 'Region', 'Order Date']).reset_index(drop=True)
        
        print(" Calculating timeline feature matrices...")


        df['sales_lag_1'] = df.groupby(['Category', 'Region'])[self.target_col].shift(1)
        df['sales_lag_2'] = df.groupby(['Category', 'Region'])[self.target_col].shift(2)
        
        
        df['sales_roll_mean_4'] = df.groupby(['Category', 'Region'])[self.target_col].transform(
            lambda x: x.shift(1).rolling(window=4).mean()
        )
        
        df['month'] = df['Order Date'].dt.month
        df['week_of_year'] = df['Order Date'].dt.isocalendar().week.astype(int)
        
        df = df.dropna().reset_index(drop=True)
        

        categorical_cols = ['Category', 'Region']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)
        
        print(f" Feature engineering complete! Total tracking matrix columns: {df.shape[1]}")
        return df

if __name__ == "__main__":
    from data_loader import SalesDataLoader
    
    print("Initializing local feature pipeline verification...")
    try:
    
        loader = SalesDataLoader()
        raw_resampled_data = loader.load_clean_resample()
        
       
        fe = FeatureEngineer()
        feature_matrix = fe.build_features(raw_resampled_data)
        
        print("\n Generated Target Feature Set Columns:")
        print(feature_matrix.columns.tolist())
        
        print("\n Comprehensive First Row Feature Look:")
        print(feature_matrix.head(1).T)  
        
    except Exception as e:
        print(f"\n Feature pipeline verification failed. Traceback error:\n{e}")