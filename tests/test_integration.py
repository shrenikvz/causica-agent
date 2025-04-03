import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestIntegration(unittest.TestCase):
    """Integration tests for the Causica Agent system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import modules
        from src.web.interface import WebInterface
        from src.agent.langchain_agent import CausalAgent
        from src.data_processing.data_processor import DataProcessor
        from src.causica_integration.causica_wrapper import CausicalWrapper
        from src.causica_integration.causica_integration import CausicalIntegration
        
        # Create components
        self.web_interface = WebInterface()
        self.data_processor = DataProcessor()
        self.causica_wrapper = CausicalWrapper()
        self.causica_integration = CausicalIntegration(wrapper=self.causica_wrapper)
        self.agent = CausalAgent(model="o3-mini")
        
        # Create a test CSV file
        self.test_csv_path = "test_integration_data.csv"
        test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [2, 3, 4, 5, 6],
            'D': [3, 1, 4, 2, 5]
        })
        test_df.to_csv(self.test_csv_path, index=False)
        
        # Load the test dataset
        self.test_df = self.data_processor.load_csv(self.test_csv_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
    
    def test_end_to_end_workflow(self):
        """Test the end-to-end workflow from data loading to causal inference"""
        # 1. Load and process the dataset
        self.assertIsInstance(self.test_df, pd.DataFrame)
        self.assertEqual(self.test_df.shape, (5, 4))
        
        # 2. Discover causal structure
        model = self.causica_wrapper.discover_causal_structure(self.test_df)
        self.assertIsInstance(model, dict)
        self.assertTrue(model.get('trained', False))
        
        # 3. Get causal graph
        graph = self.causica_wrapper.get_causal_graph(model)
        self.assertIsInstance(graph, dict)
        self.assertIn('nodes', graph)
        self.assertIn('edges', graph)
        
        # 4. Estimate treatment effect
        effect_results = self.causica_integration.estimate_treatment_effect(model, 'A', 'B')
        self.assertIsInstance(effect_results, dict)
        self.assertIn('ate', effect_results)
        self.assertIn('ate_table', effect_results)
        self.assertIn('visualization', effect_results)
        
        # 5. Perform counterfactual analysis
        cf_results = self.causica_integration.perform_counterfactual_analysis(
            model, {'A': 2.0}, ['B', 'C']
        )
        self.assertIsInstance(cf_results, dict)
        self.assertIn('predictions', cf_results)
        self.assertIn('cf_table', cf_results)
        self.assertIn('visualization', cf_results)
        
        # 6. Generate explanation of causal graph
        explanation = self.causica_integration.explain_causal_graph(graph)
        self.assertIsInstance(explanation, str)
        self.assertIn("Causal Graph Explanation", explanation)
    
    @patch('src.agent.langchain_agent.AgentExecutor')
    def test_agent_integration(self, mock_agent_executor):
        """Test the integration of the LangChain agent with other components"""
        # Mock the agent executor
        mock_executor = MagicMock()
        mock_agent_executor.return_value = mock_executor
        mock_executor.invoke.return_value = {"output": "This is a test response"}
        
        # Replace the agent's executor with our mock
        self.agent.agent_executor = mock_executor
        
        # 1. Discover causal structure
        model = self.causica_wrapper.discover_causal_structure(self.test_df)
        
        # 2. Process a message through the agent
        response = self.agent.process_message(
            "What is the causal relationship between A and B?",
            dataset=self.test_df,
            causal_model=model
        )
        
        # Check that the executor's invoke method was called
        mock_executor.invoke.assert_called_once()
        
        # Check the response
        self.assertEqual(response, "This is a test response")

if __name__ == '__main__':
    unittest.main()
