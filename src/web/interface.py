import streamlit as st
import pandas as pd
import json
import os
import sys
from typing import Dict, Any, Optional

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class WebInterface:
    """
    Class to handle web interface components and interactions
    """
    
    def __init__(self):
        """Initialize the web interface components"""
        pass
    
    def display_dataset_info(self, df: pd.DataFrame) -> None:
        """
        Display dataset information in the sidebar
        
        Args:
            df: Pandas DataFrame containing the dataset
        """
        st.sidebar.subheader("Dataset Preview")
        st.sidebar.dataframe(df.head())
        
        st.sidebar.subheader("Dataset Statistics")
        st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        st.sidebar.write("Column types:")
        st.sidebar.write(df.dtypes)
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            st.sidebar.warning("Dataset contains missing values")
            st.sidebar.write(missing_values[missing_values > 0])
    
    def display_causal_controls(self, 
                               df: pd.DataFrame, 
                               on_discover_callback: callable,
                               on_estimate_callback: callable) -> None:
        """
        Display causal discovery and inference controls
        
        Args:
            df: Pandas DataFrame containing the dataset
            on_discover_callback: Callback function for causal discovery
            on_estimate_callback: Callback function for causal effect estimation
        """
        st.sidebar.header("Causal Discovery")
        if st.sidebar.button("Discover Causal Structure"):
            with st.spinner("Discovering causal structure..."):
                try:
                    on_discover_callback(df)
                    st.sidebar.success("Causal structure discovered successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error during causal discovery: {str(e)}")
        
        st.sidebar.header("Causal Inference")
        treatment_var = st.sidebar.selectbox(
            "Select treatment variable",
            options=df.columns.tolist()
        )
        
        outcome_var = st.sidebar.selectbox(
            "Select outcome variable",
            options=df.columns.tolist()
        )
        
        if st.sidebar.button("Estimate Causal Effect"):
            with st.spinner("Estimating causal effect..."):
                try:
                    effect = on_estimate_callback(treatment_var, outcome_var)
                    st.sidebar.success(f"Average Treatment Effect: {effect:.4f}")
                except Exception as e:
                    st.sidebar.error(f"Error estimating causal effect: {str(e)}")
    
    def display_chat_interface(self, 
                              messages: list, 
                              on_message_callback: callable) -> None:
        """
        Display chat interface with message history
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            on_message_callback: Callback function for processing user messages
        """
        st.header("Chat with the Causal Agent")
        
        # Display chat messages
        for message in messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about your data or causal analysis...")
        
        if user_input:
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Process user message and get response
            with st.spinner("Thinking..."):
                response = on_message_callback(user_input)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
            
            return {"user_input": user_input, "response": response}
        
        return None
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """
        Save an uploaded file to disk
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Path to the saved file
        """
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
