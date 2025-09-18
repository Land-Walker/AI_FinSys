import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class FinanceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing financial time series data
    with macroeconomic features and scenario labels.
    """
    def __init__(self, stock_path, macro_path, label_path, sequence_length, target_col='close'):
        self.sequence_length = sequence_length
        self.target_col = target_col
        
        # --- 1. Load, Merge, and Preprocess Data ---
        stock_df = pd.read_csv(stock_path, parse_dates=['date'])
        macro_df = pd.read_csv(macro_path, parse_dates=['date'])
        label_df = pd.read_csv(label_path, parse_dates=['date'])
        
        # Merge into a single dataframe
        df = pd.merge(stock_df, macro_df, on='date', how='inner')
        df = pd.merge(df, label_df, on='date', how='inner')
        df = df.sort_values(by='date').reset_index(drop=True)
        
        # --- 2. Encode Scenario Labels ---
        self.label_mapping = {'bull': 0, 'bear': 1, 'neutral': 2}
        df['label'] = df['label'].map(self.label_mapping)
        self.labels = df['label'].values
        
        # --- 3. Normalize Features ---
        self.scaler_target = MinMaxScaler()
        self.scaler_macro = MinMaxScaler()
        
        self.macro_cols = [col for col in macro_df.columns if col != 'date']
        
        # Ensure target column exists
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in stock data.")
            
        # Fit and transform the data
        # We reshape(-1, 1) for single-feature scaling
        self.target_data = self.scaler_target.fit_transform(df[[self.target_col]])
        self.macro_data = self.scaler_macro.fit_transform(df[self.macro_cols])
        
        print("Data loaded and preprocessed successfully!")
        print(f"Total time steps: {len(self.target_data)}")
        print(f"Number of macro features: {len(self.macro_cols)}")

    def __len__(self):
        """Returns the total number of samples (windows) available."""
        return len(self.target_data) - self.sequence_length + 1

    def __getitem__(self, index):
        """
        Retrieves one sample (a window of data and its corresponding label).
        """
        start_idx = index
        end_idx = index + self.sequence_length
        
        # Slice the data to get the window
        target_window = self.target_data[start_idx:end_idx]
        macro_window = self.macro_data[start_idx:end_idx]
        
        # The label corresponds to the state at the end of the window
        label = self.labels[end_idx - 1]
        
        # Convert to PyTorch Tensors and return
        return {
            # Squeeze to remove the last dimension if it's 1
            'target': torch.tensor(target_window, dtype=torch.float32).squeeze(-1),
            'macro_features': torch.tensor(macro_window, dtype=torch.float32),
            'scenario_label': torch.tensor(label, dtype=torch.long)
        }
