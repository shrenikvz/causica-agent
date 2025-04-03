import os
import sys
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CausicalIntegration:
    """
    Integration with Microsoft Causica library for causal discovery and inference
    """
    
    def __init__(self, wrapper=None):
        """
        Initialize the Causica integration
        
        Args:
            wrapper: Optional CausicalWrapper instance
        """
        if wrapper is None:
            from causica_integration.causica_wrapper import CausicalWrapper
            self.wrapper = CausicalWrapper()
        else:
            self.wrapper = wrapper
    
    def load_and_process_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load and process a dataset from a CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Processed pandas DataFrame
        """
        try:
            # Load the dataset
            from data_processing.data_processor import DataProcessor
            processor = DataProcessor()
            df = processor.load_csv(file_path)
            
            return df
        except Exception as e:
            raise Exception(f"Error loading and processing dataset: {str(e)}")
    
    def discover_causal_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Discover causal structure in the dataset
        
        Args:
            df: Pandas DataFrame containing the dataset
            
        Returns:
            Dictionary containing the trained causal model
        """
        try:
            return self.wrapper.discover_causal_structure(df)
        except Exception as e:
            raise Exception(f"Error during causal discovery: {str(e)}")
    
    def get_causal_graph(self, model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the causal graph
        
        Args:
            model: Trained causal model (optional, uses stored model if None)
            
        Returns:
            Dictionary containing nodes and edges of the causal graph
        """
        try:
            return self.wrapper.get_causal_graph(model)
        except Exception as e:
            raise Exception(f"Error getting causal graph: {str(e)}")
    
    def estimate_treatment_effect(self, 
                                 model: Optional[Dict[str, Any]], 
                                 treatment: str, 
                                 outcome: str,
                                 num_samples: int = 1000) -> Dict[str, Any]:
        """
        Estimate the treatment effect
        
        Args:
            model: Trained causal model (optional, uses stored model if None)
            treatment: Treatment variable
            outcome: Outcome variable
            num_samples: Number of samples for estimation
            
        Returns:
            Dictionary containing treatment effect information
        """
        try:
            ate = self.wrapper.estimate_treatment_effect(model, treatment, outcome, num_samples)
            
            # Create a table for the results
            ate_table = pd.DataFrame({
                'Treatment': [treatment],
                'Outcome': [outcome],
                'Average Treatment Effect': [f"{ate:.4f}"],
                'Samples': [num_samples]
            })
            
            # Create a simple visualization (in a real implementation, this would be more sophisticated)
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[treatment],
                y=[ate],
                text=[f"{ate:.4f}"],
                textposition='auto',
                marker_color='royalblue'
            ))
            
            fig.update_layout(
                title=f"Average Treatment Effect of {treatment} on {outcome}",
                xaxis_title="Treatment Variable",
                yaxis_title="Effect Size",
                yaxis=dict(
                    zeroline=True,
                    zerolinewidth=2,
                    zerolinecolor='black'
                )
            )
            
            return {
                'ate': ate,
                'ate_table': ate_table,
                'visualization': fig,
                'treatment': treatment,
                'outcome': outcome,
                'num_samples': num_samples
            }
        except Exception as e:
            raise Exception(f"Error estimating treatment effect: {str(e)}")
    
    def perform_counterfactual_analysis(self,
                                       model: Optional[Dict[str, Any]],
                                       intervention: Dict[str, float],
                                       target_variables: Optional[List[str]] = None,
                                       num_samples: int = 1000) -> Dict[str, Any]:
        """
        Perform counterfactual analysis
        
        Args:
            model: Trained causal model (optional, uses stored model if None)
            intervention: Dictionary mapping variable names to intervention values
            target_variables: List of target variables to predict (optional, all if None)
            num_samples: Number of samples for estimation
            
        Returns:
            Dictionary containing counterfactual analysis results
        """
        try:
            results = self.wrapper.perform_counterfactual_analysis(
                model, intervention, target_variables, num_samples
            )
            
            # Create a table for the results
            predictions = results['predictions']
            cf_table = pd.DataFrame({
                'Variable': list(predictions.keys()),
                'Predicted Value': [f"{val:.4f}" for val in predictions.values()]
            })
            
            # Add intervention information
            intervention_str = ", ".join([f"{var}={val:.4f}" for var, val in intervention.items()])
            
            # Create a simple visualization (in a real implementation, this would be more sophisticated)
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(predictions.keys()),
                y=list(predictions.values()),
                text=[f"{val:.4f}" for val in predictions.values()],
                textposition='auto',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title=f"Counterfactual Predictions with Intervention: {intervention_str}",
                xaxis_title="Variables",
                yaxis_title="Predicted Values"
            )
            
            return {
                'predictions': predictions,
                'cf_table': cf_table,
                'visualization': fig,
                'intervention': intervention,
                'num_samples': num_samples
            }
        except Exception as e:
            raise Exception(f"Error performing counterfactual analysis: {str(e)}")
    
    def explain_causal_graph(self, graph: Dict[str, Any]) -> str:
        """
        Generate an explanation of the causal graph
        
        Args:
            graph: Dictionary containing nodes and edges of the causal graph
            
        Returns:
            String explanation of the causal graph
        """
        try:
            nodes = graph['nodes']
            edges = graph['edges']
            
            # Count incoming and outgoing edges for each node
            incoming = {node: 0 for node in nodes}
            outgoing = {node: 0 for node in nodes}
            
            for edge in edges:
                source = edge['source']
                target = edge['target']
                outgoing[source] += 1
                incoming[target] += 1
            
            # Identify root causes (no incoming edges)
            root_causes = [node for node in nodes if incoming[node] == 0 and outgoing[node] > 0]
            
            # Identify leaf effects (no outgoing edges)
            leaf_effects = [node for node in nodes if outgoing[node] == 0 and incoming[node] > 0]
            
            # Identify mediators (both incoming and outgoing edges)
            mediators = [node for node in nodes if incoming[node] > 0 and outgoing[node] > 0]
            
            # Generate explanation
            explanation = "# Causal Graph Explanation\n\n"
            
            explanation += f"The causal graph contains {len(nodes)} variables and {len(edges)} causal relationships.\n\n"
            
            if root_causes:
                explanation += "## Root Causes\n\n"
                explanation += "These variables appear to be root causes (they influence other variables but are not influenced by any variables in the dataset):\n"
                for node in root_causes:
                    explanation += f"- {node} (influences {outgoing[node]} other variables)\n"
                explanation += "\n"
            
            if leaf_effects:
                explanation += "## Outcome Variables\n\n"
                explanation += "These variables appear to be final outcomes (they are influenced by other variables but don't influence any variables in the dataset):\n"
                for node in leaf_effects:
                    explanation += f"- {node} (influenced by {incoming[node]} other variables)\n"
                explanation += "\n"
            
            if mediators:
                explanation += "## Mediator Variables\n\n"
                explanation += "These variables act as mediators (they are both influenced by some variables and influence other variables):\n"
                for node in mediators:
                    explanation += f"- {node} (influenced by {incoming[node]} variables and influences {outgoing[node]} variables)\n"
                explanation += "\n"
            
            explanation += "## Key Causal Relationships\n\n"
            
            # Sort edges by weight to find strongest relationships
            sorted_edges = sorted(edges, key=lambda e: e.get('weight', 0), reverse=True)
            top_edges = sorted_edges[:min(5, len(sorted_edges))]
            
            if top_edges:
                explanation += "The strongest causal relationships in the graph are:\n"
                for edge in top_edges:
                    explanation += f"- {edge['source']} → {edge['target']} (strength: {edge.get('weight', 0):.2f})\n"
            
            return explanation
        except Exception as e:
            raise Exception(f"Error explaining causal graph: {str(e)}")
