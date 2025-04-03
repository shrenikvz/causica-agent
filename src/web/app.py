import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import os
import sys
import json
from typing import List, Dict, Any, Optional

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from other modules (to be implemented)
from agent.langchain_agent import CausalAgent
from data_processing.data_processor import DataProcessor
from causica_integration.causica_wrapper import CausicalWrapper

# Initialize session state variables if they don't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'causal_model' not in st.session_state:
    st.session_state.causal_model = None

if 'dataset' not in st.session_state:
    st.session_state.dataset = None

if 'causal_graph' not in st.session_state:
    st.session_state.causal_graph = None

if 'agent' not in st.session_state:
    # Initialize the agent (will be implemented in agent module)
    st.session_state.agent = CausalAgent(model="o3-mini")

if 'data_processor' not in st.session_state:
    # Initialize the data processor (will be implemented in data_processing module)
    st.session_state.data_processor = DataProcessor()

if 'causica_wrapper' not in st.session_state:
    # Initialize the Causica wrapper (will be implemented in causica_integration module)
    st.session_state.causica_wrapper = CausicalWrapper()

def visualize_causal_graph(graph_data: Dict[str, Any]) -> go.Figure:
    """
    Create a plotly figure to visualize the causal graph
    
    Args:
        graph_data: Dictionary containing nodes and edges of the causal graph
        
    Returns:
        Plotly figure object
    """
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_data['nodes']:
        G.add_node(node)
    
    # Add edges
    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
    
    # Create positions for nodes
    pos = nx.spring_layout(G)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = edge[2].get('weight', 1.0)
        edge_text.append(f"Weight: {weight:.2f}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='#007BFF',
            size=15,
            line=dict(width=2, color='#FFFFFF')))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Causal Graph',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(255,255,255,1)',
                        paper_bgcolor='rgba(255,255,255,1)',
                    ))
    
    return fig

def process_user_message(user_message: str) -> str:
    """
    Process user message using the LangChain agent
    
    Args:
        user_message: User's message text
        
    Returns:
        Agent's response
    """
    # Use the agent to process the message (will be implemented in agent module)
    response = st.session_state.agent.process_message(
        user_message, 
        dataset=st.session_state.dataset,
        causal_model=st.session_state.causal_model
    )
    return response

def main():
    st.set_page_config(
        page_title="Causal Inference Agent",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Causal Inference Agent")
    
    # Sidebar for dataset upload and model controls
    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Process the uploaded dataset
                df = pd.read_csv(uploaded_file)
                st.session_state.dataset = df
                st.success(f"Dataset loaded: {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
                
                # Display dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                # Dataset statistics
                st.subheader("Dataset Statistics")
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                st.write("Column types:")
                st.write(df.dtypes)
                
                # Missing values
                missing_values = df.isnull().sum()
                if missing_values.sum() > 0:
                    st.warning("Dataset contains missing values")
                    st.write(missing_values[missing_values > 0])
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
        
        st.header("Causal Discovery")
        if st.session_state.dataset is not None:
            if st.button("Discover Causal Structure"):
                with st.spinner("Discovering causal structure..."):
                    # This will be implemented in the causica_integration module
                    try:
                        # Placeholder for causal discovery
                        st.session_state.causal_model = st.session_state.causica_wrapper.discover_causal_structure(
                            st.session_state.dataset
                        )
                        st.session_state.causal_graph = st.session_state.causica_wrapper.get_causal_graph(
                            st.session_state.causal_model
                        )
                        st.success("Causal structure discovered successfully!")
                    except Exception as e:
                        st.error(f"Error during causal discovery: {str(e)}")
        
        st.header("Causal Inference")
        if st.session_state.causal_model is not None:
            st.subheader("Treatment Variable")
            treatment_var = st.selectbox(
                "Select treatment variable",
                options=st.session_state.dataset.columns.tolist()
            )
            
            st.subheader("Outcome Variable")
            outcome_var = st.selectbox(
                "Select outcome variable",
                options=st.session_state.dataset.columns.tolist()
            )
            
            if st.button("Estimate Causal Effect"):
                with st.spinner("Estimating causal effect..."):
                    # This will be implemented in the causica_integration module
                    try:
                        effect = st.session_state.causica_wrapper.estimate_treatment_effect(
                            st.session_state.causal_model,
                            treatment_var,
                            outcome_var
                        )
                        st.success(f"Average Treatment Effect: {effect:.4f}")
                    except Exception as e:
                        st.error(f"Error estimating causal effect: {str(e)}")
    
    # Main area with chat interface and visualization
    col1, col2 = st.columns([3, 2])
    
    # Chat interface in the left column
    with col1:
        st.header("Chat with the Causal Agent")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask a question about your data or causal analysis...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Process user message and get response
            with st.spinner("Thinking..."):
                response = process_user_message(user_input)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
    
    # Visualization in the right column
    with col2:
        st.header("Causal Graph Visualization")
        
        if st.session_state.causal_graph is not None:
            # Visualize the causal graph
            fig = visualize_causal_graph(st.session_state.causal_graph)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No causal graph available. Upload a dataset and discover causal structure to visualize.")

if __name__ == "__main__":
    main()
