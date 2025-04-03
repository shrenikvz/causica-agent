import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from src.web.interface import WebInterface
from src.agent.langchain_agent import CausalAgent
from src.data_processing.data_processor import DataProcessor
from src.causica_integration.causica_wrapper import CausicalWrapper
from src.causica_integration.causica_integration import CausicalIntegration

class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        
        # Create a test CSV file
        self.test_csv_path = "test_data.csv"
        test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        test_df.to_csv(self.test_csv_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)
    
    def test_load_csv(self):
        """Test loading a CSV file"""
        df = self.processor.load_csv(self.test_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 3))
        self.assertEqual(list(df.columns), ['A', 'B', 'C'])
    
    def test_preprocess_for_causica(self):
        """Test preprocessing data for Causica"""
        df = self.processor.load_csv(self.test_csv_path)
        causica_data = self.processor.preprocess_for_causica(df)
        
        self.assertIsInstance(causica_data, dict)
        self.assertIn('data', causica_data)
        self.assertIn('variables_metadata', causica_data)
        
        # Check that variables metadata contains all columns
        for col in df.columns:
            self.assertIn(col, causica_data['variables_metadata'])

class TestCausicalWrapper(unittest.TestCase):
    """Test cases for the CausicalWrapper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.wrapper = CausicalWrapper()
        
        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [2, 3, 4, 5, 6]
        })
    
    def test_discover_causal_structure(self):
        """Test discovering causal structure"""
        model = self.wrapper.discover_causal_structure(self.test_df)
        
        self.assertIsInstance(model, dict)
        self.assertIn('type', model)
        self.assertEqual(model['type'], 'DECI')
        self.assertIn('variables', model)
        self.assertIn('adjacency_matrix', model)
        self.assertTrue(model['trained'])
    
    def test_get_causal_graph(self):
        """Test getting causal graph"""
        # First discover causal structure
        model = self.wrapper.discover_causal_structure(self.test_df)
        
        # Then get the causal graph
        graph = self.wrapper.get_causal_graph(model)
        
        self.assertIsInstance(graph, dict)
        self.assertIn('nodes', graph)
        self.assertIn('edges', graph)
        self.assertEqual(len(graph['nodes']), 3)  # A, B, C
    
    def test_estimate_treatment_effect(self):
        """Test estimating treatment effect"""
        # First discover causal structure
        model = self.wrapper.discover_causal_structure(self.test_df)
        
        # Then estimate treatment effect
        effect = self.wrapper.estimate_treatment_effect(model, 'A', 'B')
        
        self.assertIsInstance(effect, float)
    
    def test_perform_counterfactual_analysis(self):
        """Test performing counterfactual analysis"""
        # First discover causal structure
        model = self.wrapper.discover_causal_structure(self.test_df)
        
        # Then perform counterfactual analysis
        results = self.wrapper.perform_counterfactual_analysis(
            model, {'A': 2.0}, ['B', 'C']
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('predictions', results)
        self.assertIn('intervention', results)
        self.assertIn('B', results['predictions'])
        self.assertIn('C', results['predictions'])

class TestCausicalIntegration(unittest.TestCase):
    """Test cases for the CausicalIntegration class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock the CausicalWrapper
        self.mock_wrapper = MagicMock()
        self.integration = CausicalIntegration(wrapper=self.mock_wrapper)
        
        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [2, 3, 4, 5, 6]
        })
        
        # Mock the wrapper's methods
        self.mock_wrapper.discover_causal_structure.return_value = {
            'type': 'DECI',
            'variables': ['A', 'B', 'C'],
            'adjacency_matrix': np.array([
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0]
            ]),
            'trained': True
        }
        
        self.mock_wrapper.get_causal_graph.return_value = {
            'nodes': ['A', 'B', 'C'],
            'edges': [
                {'source': 'A', 'target': 'B', 'weight': 0.8},
                {'source': 'A', 'target': 'C', 'weight': 0.6},
                {'source': 'B', 'target': 'C', 'weight': 0.4}
            ]
        }
        
        self.mock_wrapper.estimate_treatment_effect.return_value = 0.5
        
        self.mock_wrapper.perform_counterfactual_analysis.return_value = {
            'predictions': {'B': 3.0, 'C': 4.0},
            'intervention': {'A': 2.0},
            'num_samples': 1000
        }
    
    def test_discover_causal_structure(self):
        """Test discovering causal structure"""
        model = self.integration.discover_causal_structure(self.test_df)
        
        # Check that the wrapper's method was called
        self.mock_wrapper.discover_causal_structure.assert_called_once_with(self.test_df)
        
        # Check the returned model
        self.assertEqual(model['type'], 'DECI')
        self.assertEqual(model['variables'], ['A', 'B', 'C'])
        self.assertTrue(model['trained'])
    
    def test_get_causal_graph(self):
        """Test getting causal graph"""
        graph = self.integration.get_causal_graph()
        
        # Check that the wrapper's method was called
        self.mock_wrapper.get_causal_graph.assert_called_once_with(None)
        
        # Check the returned graph
        self.assertEqual(graph['nodes'], ['A', 'B', 'C'])
        self.assertEqual(len(graph['edges']), 3)
    
    def test_estimate_treatment_effect(self):
        """Test estimating treatment effect"""
        results = self.integration.estimate_treatment_effect(None, 'A', 'B')
        
        # Check that the wrapper's method was called
        self.mock_wrapper.estimate_treatment_effect.assert_called_once_with(None, 'A', 'B', 1000)
        
        # Check the returned results
        self.assertEqual(results['ate'], 0.5)
        self.assertIn('ate_table', results)
        self.assertIn('visualization', results)
    
    def test_explain_causal_graph(self):
        """Test explaining causal graph"""
        graph = self.mock_wrapper.get_causal_graph.return_value
        explanation = self.integration.explain_causal_graph(graph)
        
        self.assertIsInstance(explanation, str)
        self.assertIn("Causal Graph Explanation", explanation)
        self.assertIn("Root Causes", explanation)
        self.assertIn("A", explanation)  # A is a root cause

class TestLangChainAgent(unittest.TestCase):
    """Test cases for the CausalAgent class"""
    
    @patch('src.agent.langchain_agent.ChatOpenAI')
    def setUp(self, mock_chat_openai):
        """Set up test fixtures"""
        # Mock the ChatOpenAI class
        self.mock_llm = MagicMock()
        mock_chat_openai.return_value = self.mock_llm
        
        # Create the agent
        self.agent = CausalAgent(model="o3-mini")
        
        # Create a test DataFrame
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 4, 3, 2, 1],
            'C': [2, 3, 4, 5, 6]
        })
    
    @patch('src.agent.langchain_agent.AgentExecutor')
    def test_process_message(self, mock_agent_executor):
        """Test processing a message"""
        # Mock the agent executor
        mock_executor = MagicMock()
        mock_agent_executor.return_value = mock_executor
        mock_executor.invoke.return_value = {"output": "This is a test response"}
        
        # Replace the agent's executor with our mock
        self.agent.agent_executor = mock_executor
        
        # Process a message
        response = self.agent.process_message(
            "What is the causal relationship between A and B?",
            dataset=self.test_df,
            causal_model={"type": "DECI"}
        )
        
        # Check that the executor's invoke method was called
        mock_executor.invoke.assert_called_once()
        
        # Check the response
        self.assertEqual(response, "This is a test response")

if __name__ == '__main__':
    unittest.main()
