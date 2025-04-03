from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import Dict, Any, List, Optional
import os
import sys
import pandas as pd
import streamlit as st  # Import streamlit

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
        # Retrieve the API key from Streamlit secrets
        openai_api_key = st.secrets.get("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API key not found in Streamlit secrets. Please add it.")
            # Optionally, you could raise an exception or handle this case differently
            return 

        self.model = model
        # Pass the API key to the ChatOpenAI client
        self.llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
        
        # Updated memory setup
        self.store: Dict[str, BaseChatMessageHistory] = {} # Store for session histories
        self.memory = self._get_session_history("default_session") # Get default history

        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
    
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Gets or creates a chat history for a given session ID."""
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def _create_tools(self) -> List[BaseTool]:
        """
        Create tools for the agent to use
        
        Returns:
            List of LangChain tools
        """
        # Define tools for causal inference tasks
        class DatasetInfoTool(BaseTool):
            name: str = "dataset_info"
            description: str = "Get information about the current dataset"
            
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
            name: str = "causal_discovery"
            description: str = "Discover causal structure in the dataset"
            
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
            name: str = "causal_effect"
            description: str = "Estimate causal effect between treatment and outcome variables"
            
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
            name: str = "explain_causal_graph"
            description: str = "Explain the discovered causal graph"
            
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
        
        # Updated AgentExecutor initialization to use RunnableWithMessageHistory
        # Note: This assumes you might want session-specific memory later.
        # If you ONLY need one global history, the original ConversationBufferMemory was simpler.
        # However, this pattern is more flexible for multi-user/session scenarios.
        agent_with_chat_history = RunnableWithMessageHistory(
            agent, # type: ignore
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        return AgentExecutor(
            agent=agent_with_chat_history, # type: ignore
            tools=self.tools,
            # memory is now handled by RunnableWithMessageHistory
            verbose=True,
            handle_parsing_errors=True
        )
    
    def process_message(self, message: str, session_id: str = "default_session", dataset: Optional[pd.DataFrame] = None, causal_model: Optional[Any] = None) -> str:
        """
        Process a user message and return the agent's response
        
        Args:
            message: User's message
            session_id: Identifier for the conversation session
            dataset: Current dataset (if any)
            causal_model: Current causal model (if any)
            
        Returns:
            Agent's response
        """
        # Make dataset and model available to tools
        globals()['dataset'] = dataset
        globals()['causal_model'] = causal_model
        
        try:
            # Updated invocation to include config for session_id
            response = self.agent_executor.invoke(
                {"input": message},
                config={"configurable": {"session_id": session_id}}
            )
            return response["output"]
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}"
