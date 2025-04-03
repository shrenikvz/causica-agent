import os
import sys
import pandas as pd
from typing import Dict, Any, Optional, List

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataProcessor:
    """
    Class to handle data processing for causal inference
    """
    
    def __init__(self):
        """Initialize the data processor"""
        self.current_dataset = None
        self.dataset_metadata = None
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Pandas DataFrame containing the dataset
        """
        try:
            df = pd.read_csv(file_path)
            self.current_dataset = df
            self._generate_metadata(df)
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV file: {str(e)}")
    
    def _generate_metadata(self, df: pd.DataFrame) -> None:
        """
        Generate metadata for the dataset
        
        Args:
            df: Pandas DataFrame containing the dataset
        """
        metadata = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": {col: int(count) for col, count in df.isnull().sum().items() if count > 0},
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Add basic statistics for numeric columns
        if metadata["numeric_columns"]:
            metadata["numeric_stats"] = df[metadata["numeric_columns"]].describe().to_dict()
        
        # Add value counts for categorical columns (limited to top 10)
        if metadata["categorical_columns"]:
            metadata["categorical_stats"] = {
                col: df[col].value_counts().head(10).to_dict() 
                for col in metadata["categorical_columns"]
            }
        
        self.dataset_metadata = metadata
    
    def preprocess_for_causica(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Preprocess the dataset for Causica
        
        Args:
            df: Pandas DataFrame containing the dataset
            
        Returns:
            Dictionary containing preprocessed data in Causica-compatible format
        """
        # Handle missing values
        df_processed = df.copy()
        
        # For numeric columns, fill missing values with mean
        numeric_cols = df_processed.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
        
        # For categorical columns, fill missing values with mode
        cat_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # Convert categorical variables to one-hot encoding
        if len(cat_cols) > 0:
            df_processed = pd.get_dummies(df_processed, columns=list(cat_cols))
        
        # Normalize numeric columns
        for col in numeric_cols:
            if df_processed[col].std() > 0:  # Avoid division by zero
                df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
        
        # Create variables metadata for Causica
        variables_metadata = {}
        for col in df_processed.columns:
            variables_metadata[col] = {
                "type": "continuous" if col in numeric_cols else "binary",
                "lower": float(df_processed[col].min()) if col in numeric_cols else 0,
                "upper": float(df_processed[col].max()) if col in numeric_cols else 1
            }
        
        # Prepare data in Causica-compatible format
        causica_data = {
            "data": df_processed.to_dict(orient="list"),
            "variables_metadata": variables_metadata
        }
        
        return causica_data
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current dataset
        
        Returns:
            Dictionary containing dataset summary information
        """
        if self.current_dataset is None or self.dataset_metadata is None:
            return {"error": "No dataset loaded"}
        
        return {
            "shape": self.dataset_metadata["shape"],
            "columns": self.dataset_metadata["columns"],
            "missing_values": self.dataset_metadata["missing_values"],
            "numeric_columns": self.dataset_metadata["numeric_columns"],
            "categorical_columns": self.dataset_metadata["categorical_columns"]
        }
