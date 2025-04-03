 from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from typing import Dict, Any, List, Optional
import os
import sys
import pandas as pd

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CausalAgent:
    """
    LangChain agent for causal inference using OpenAI's o3-mini model
    """
    
    def __init__(self, model: str = "o3-mini"):
        """
        Initialize the causal agent
        
        Args:
            model: OpenAI model to use (default: o3-mini)
        """
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List[BaseTool]:
        """
        Create tools for the agent to use
        
        Returns:
            List of LangChain tools
        """
        # Define tools for causal inference tasks
        class DatasetInfoTool(BaseTool):
            name = "dataset_info"
            description = "Get information about the current dataset"
            
            def _run(self, query: str) -> str:
                if 'dataset' not in globals() or globals()['dataset'] is None:
                    return "No dataset is currently loaded. Please upload a CSV file first."
                
                df = globals()['dataset']
                info = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_values": {col: int(count) for col, count in df.isnull().sum().items() if count > 0},
                }
                return str(info)
        
        class CausalDiscoveryTool(BaseTool):
            name = "causal_discovery"
            description = "Discover causal structure in the dataset"
            
            def _run(self, query: str) -> str:
                if 'dataset' not in globals() or globals()['dataset'] is None:
                    return "No dataset is currently loaded. Please upload a CSV file first."
                
                if 'causica_wrapper' not in globals() or globals()['causica_wrapper'] is None:
                    return "Causica wrapper is not initialized."
                
                try:
                    # This would be implemented in the causica_integration module
                    result = "Causal discovery initiated. This will analyze the dataset to identify potential causal relationships between variables."
                    return result
                except Exception as e:
                    return f"Error during causal discovery: {str(e)}"
        
        class CausalEffectTool(BaseTool):
            name = "causal_effect"
            description = "Estimate causal effect between treatment and outcome variables"
            
            def _run(self, query: str) -> str:
                if 'dataset' not in globals() or globals()['dataset'] is None:
                    return "No dataset is currently loaded. Please upload a CSV file first."
                
                if 'causal_model' not in globals() or globals()['causal_model'] is None:
                    return "No causal model is available. Please run causal discovery first."
                
                # Extract treatment and outcome variables from query
                # This is a simplified implementation
                import re
                treatment_match = re.search(r"treatment[:\s]+([^\s,]+)", query, re.IGNORECASE)
                outcome_match = re.search(r"outcome[:\s]+([^\s,]+)", query, re.IGNORECASE)
                
                if not treatment_match or not outcome_match:
                    return "Please specify both treatment and outcome variables."
                
                treatment = treatment_match.group(1)
                outcome = outcome_match.group(1)
                
                try:
                    # This would be implemented in the causica_integration module
                    result = f"Estimating causal effect of {treatment} on {outcome}. This will calculate how changes in {treatment} causally affect {outcome}."
                    return result
                except Exception as e:
                    return f"Error estimating causal effect: {str(e)}"
        
        class ExplainCausalGraphTool(BaseTool):
            name = "explain_causal_graph"
            description = "Explain the discovered causal graph"
            
            def _run(self, query: str) -> str:
                if 'causal_graph' not in globals() or globals()['causal_graph'] is None:
                    return "No causal graph is available. Please run causal discovery first."
                
                try:
                    # This would be implemented in the causica_integration module
                    result = "The causal graph shows the relationships between variables in your dataset. " \
                             "Arrows indicate the direction of causality, from cause to effect. " \
                             "The graph can help you understand which variables directly influence others."
                    return result
                except Exception as e:
                    return f"Error explaining causal graph: {str(e)}"
        
        return [
            DatasetInfoTool(),
            CausalDiscoveryTool(),
            CausalEffectTool(),
            ExplainCausalGraphTool()
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create the LangChain agent
        
        Returns:
            AgentExecutor instance
        """
        system_message = """You are an expert AI assistant specializing in causal inference and analysis.
Your role is to help users understand causal relationships in their data using the Microsoft Causica library.

When interacting with users:
1. Explain your understanding of their request before taking action
2. Provide clear explanations of causal concepts
3. Guide users through the causal inference process
4. Interpret results in plain language
5. Seek approval at key decision points

You have access to tools that can:
- Get information about the current dataset
- Discover causal structure in data
- Estimate causal effects between variables
- Explain causal graphs

Always explain what you're doing and why, and provide interpretations of results that non-experts can understand.
"""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_message(self, message: str, dataset: Optional[pd.DataFrame] = None, causal_model: Optional[Any] = None) -> str:
        """
        Process a user message and return the agent's response
        
        Args:
            message: User's message
            dataset: Current dataset (if any)
            causal_model: Current causal model (if any)
            
        Returns:
            Agent's response
        """
        # Make dataset and model available to tools
        globals()['dataset'] = dataset
        globals()['causal_model'] = causal_model
        
        try:
            response = self.agent_executor.invoke({"input": message})
            return response["output"]
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
