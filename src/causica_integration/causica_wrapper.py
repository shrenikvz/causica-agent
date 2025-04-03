import os
import sys
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CausicalWrapper:
    """
    Wrapper for Microsoft Causica library for causal discovery and inference
    """
    
    def __init__(self):
        """Initialize the Causica wrapper"""
        self.model = None
        self.graph = None
        self.dataset = None
        self.variables_metadata = None
    
    def discover_causal_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Discover causal structure in the dataset using Causica's DECI model
        
        Args:
            df: Pandas DataFrame containing the dataset
            
        Returns:
            Dictionary containing the trained causal model
        """
        try:
            # Store the dataset
            self.dataset = df
            
            # Process the dataset for Causica
            processed_data, self.variables_metadata = self._preprocess_dataset(df)
            
            # Create a DECI model
            # This is a simplified implementation that would be replaced with actual Causica code
            model = self._create_deci_model(processed_data, self.variables_metadata)
            
            # Train the model
            trained_model = self._train_model(model, processed_data)
            
            # Extract the causal graph
            self.graph = self._extract_causal_graph(trained_model)
            
            # Store the model
            self.model = trained_model
            
            return trained_model
        except Exception as e:
            raise Exception(f"Error during causal discovery: {str(e)}")
    
    def _preprocess_dataset(self, df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Preprocess the dataset for Causica
        
        Args:
            df: Pandas DataFrame containing the dataset
            
        Returns:
            Tuple of (processed_data, variables_metadata)
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
        
        # Convert to TensorDict format for Causica
        # This is a simplified version - actual implementation would use TensorDict
        processed_data = {
            "data": torch.tensor(df_processed.values, dtype=torch.float32),
            "mask": torch.ones_like(torch.tensor(df_processed.values, dtype=torch.float32)),
            "column_names": list(df_processed.columns)
        }
        
        return processed_data, variables_metadata
    
    def _create_deci_model(self, processed_data: Dict[str, Any], variables_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a DECI model
        
        Args:
            processed_data: Processed data in Causica format
            variables_metadata: Metadata for variables
            
        Returns:
            DECI model
        """
        # This is a simplified implementation that would be replaced with actual Causica code
        # In a real implementation, this would use Causica's DECI model
        
        # Placeholder for DECI model
        model = {
            "type": "DECI",
            "num_variables": len(variables_metadata),
            "variables": list(variables_metadata.keys()),
            "variables_metadata": variables_metadata
        }
        
        return model
    
    def _train_model(self, model: Dict[str, Any], processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the DECI model
        
        Args:
            model: DECI model
            processed_data: Processed data in Causica format
            
        Returns:
            Trained model
        """
        # This is a simplified implementation that would be replaced with actual Causica code
        # In a real implementation, this would use Causica's training functionality
        
        # Simulate training by adding random adjacency matrix
        num_vars = model["num_variables"]
        
        # Create a random DAG (upper triangular matrix for acyclicity)
        np.random.seed(42)  # For reproducibility
        adj_matrix = np.random.rand(num_vars, num_vars) < 0.3  # 30% chance of edge
        adj_matrix = np.triu(adj_matrix, k=1)  # Upper triangular to ensure DAG
        
        # Add the adjacency matrix to the model
        model["adjacency_matrix"] = adj_matrix
        model["trained"] = True
        
        return model
    
    def _extract_causal_graph(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the causal graph from the trained model
        
        Args:
            model: Trained DECI model
            
        Returns:
            Dictionary containing nodes and edges of the causal graph
        """
        # Extract nodes and edges from the adjacency matrix
        variables = model["variables"]
        adj_matrix = model["adjacency_matrix"]
        
        nodes = variables
        edges = []
        
        # Convert adjacency matrix to edges
        for i in range(len(variables)):
            for j in range(len(variables)):
                if adj_matrix[i, j]:
                    edges.append({
                        "source": variables[i],
                        "target": variables[j],
                        "weight": np.random.rand()  # Simulate edge weights
                    })
        
        graph = {
            "nodes": nodes,
            "edges": edges
        }
        
        return graph
    
    def get_causal_graph(self, model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the causal graph
        
        Args:
            model: Trained DECI model (optional, uses stored model if None)
            
        Returns:
            Dictionary containing nodes and edges of the causal graph
        """
        if model is None:
            if self.graph is None:
                raise Exception("No causal graph available. Please run causal discovery first.")
            return self.graph
        
        return self._extract_causal_graph(model)
    
    def estimate_treatment_effect(self, 
                                 model: Optional[Dict[str, Any]], 
                                 treatment: str, 
                                 outcome: str,
                                 num_samples: int = 1000) -> float:
        """
        Estimate the average treatment effect
        
        Args:
            model: Trained DECI model (optional, uses stored model if None)
            treatment: Treatment variable
            outcome: Outcome variable
            num_samples: Number of samples for estimation
            
        Returns:
            Estimated average treatment effect
        """
        if model is None:
            if self.model is None:
                raise Exception("No causal model available. Please run causal discovery first.")
            model = self.model
        
        # Check if variables exist in the model
        if treatment not in model["variables"] or outcome not in model["variables"]:
            raise Exception(f"Treatment or outcome variable not found in the model.")
        
        # This is a simplified implementation that would be replaced with actual Causica code
        # In a real implementation, this would use Causica's treatment effect estimation
        
        # Simulate treatment effect estimation
        # In reality, this would involve interventional sampling from the model
        treatment_idx = model["variables"].index(treatment)
        outcome_idx = model["variables"].index(outcome)
        
        # Check if there's a path from treatment to outcome in the adjacency matrix
        # This is a very simplified check
        if model["adjacency_matrix"][treatment_idx, outcome_idx]:
            # Direct effect
            effect = np.random.normal(0.5, 0.1)  # Simulate a moderate positive effect
        else:
            # Check for indirect effects (very simplified)
            has_indirect_effect = False
            for i in range(len(model["variables"])):
                if (model["adjacency_matrix"][treatment_idx, i] and 
                    model["adjacency_matrix"][i, outcome_idx]):
                    has_indirect_effect = True
                    break
            
            if has_indirect_effect:
                effect = np.random.normal(0.3, 0.1)  # Simulate a smaller indirect effect
            else:
                effect = np.random.normal(0.0, 0.05)  # Simulate no effect
        
        return float(effect)
    
    def perform_counterfactual_analysis(self,
                                       model: Optional[Dict[str, Any]],
                                       intervention: Dict[str, float],
                                       target_variables: Optional[List[str]] = None,
                                       num_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform counterfactual analysis
        
        Args:
            model: Trained DECI model (optional, uses stored model if None)
            intervention: Dictionary mapping variable names to intervention values
            target_variables: List of target variables to predict (optional, all if None)
            num_samples: Number of samples for estimation
            
        Returns:
            Dictionary containing counterfactual predictions
        """
        if model is None:
            if self.model is None:
                raise Exception("No causal model available. Please run causal discovery first.")
            model = self.model
        
        # Check if intervention variables exist in the model
        for var in intervention.keys():
            if var not in model["variables"]:
                raise Exception(f"Intervention variable {var} not found in the model.")
        
        # Set target variables to all if None
        if target_variables is None:
            target_variables = model["variables"]
        else:
            # Check if target variables exist in the model
            for var in target_variables:
                if var not in model["variables"]:
                    raise Exception(f"Target variable {var} not found in the model.")
        
        # This is a simplified implementation that would be replaced with actual Causica code
        # In a real implementation, this would use Causica's counterfactual analysis
        
        # Simulate counterfactual predictions
        # In reality, this would involve interventional sampling from the model
        predictions = {}
        
        for var in target_variables:
            if var in intervention:
                # If the variable is intervened on, the prediction is the intervention value
                predictions[var] = intervention[var]
            else:
                # Otherwise, simulate a prediction
                # This is a very simplified simulation
                var_idx = model["variables"].index(var)
                
                # Check if any intervention variable directly affects this variable
                has_direct_effect = False
                for int_var in intervention.keys():
                    int_idx = model["variables"].index(int_var)
                    if model["adjacency_matrix"][int_idx, var_idx]:
                        has_direct_effect = True
                        break
                
                if has_direct_effect:
                    # Simulate a change due to intervention
                    base_value = np.random.normal(0.0, 1.0)  # Baseline value
                    change = np.random.normal(0.5, 0.2)  # Simulated change
                    predictions[var] = base_value + change
                else:
                    # No direct effect, minimal change
                    predictions[var] = np.random.normal(0.0, 1.0)
        
        return {
            "predictions": predictions,
            "intervention": intervention,
            "num_samples": num_samples
        }
