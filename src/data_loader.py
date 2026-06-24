import pandas as pd
import yaml
import os

class SalesDataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the engine and reads your central control panel configuration.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f" Configuration file not found at: {config_path}")
            
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.raw_path = self.config['paths']['raw_data']
        self.date_col = self.config['columns']['date']
        self.target_col = self.config['columns']['target']
        self.cadence = self.config['pipeline']['resample_cadence']

    def load_clean_resample(self) -> pd.DataFrame:
        """
        Loads raw data, preserves important business columns, and resamples to weeks.
        """
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f" Raw dataset missing at: {self.raw_path}")
            
        print(f" Loading raw dataset from: {self.raw_path}")
        df = pd.read_csv(self.raw_path, encoding='latin1')
        
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors='coerce')
        df = df.dropna(subset=[self.date_col, self.target_col])
        
        important_cols = [self.date_col, self.target_col, 'Profit', 'Discount', 'Category', 'Region']
        df = df[important_cols]
        
        daily_grouped = df.groupby([self.date_col, 'Category', 'Region']).agg({
            self.target_col: 'sum',   
            'Profit': 'sum',          
            'Discount': 'mean'        
        }).reset_index()
        
       
        daily_grouped.set_index(self.date_col, inplace=True)
        
        weekly_df = daily_grouped.groupby(['Category', 'Region']).resample(self.cadence).agg({
            self.target_col: 'sum',
            'Profit': 'sum',
            'Discount': 'mean'
        }).fillna(0).reset_index()
        
        print(f" Success: Formatted data into multi-dimensional business weeks.")
        return weekly_df

if __name__ == "__main__":
    print(" Running internal pipeline verification check...")
    try:
        loader = SalesDataLoader()
        test_data = loader.load_clean_resample()
        print("\n Preview of your rich, high-quality feature matrix:")
        print(test_data.head())
    except Exception as e:
        print(f"\n Pipeline test failed. Error detail:\n{e}")