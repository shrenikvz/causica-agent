import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, Any

def create_causal_graph_visualization(graph_data: Dict[str, Any]) -> go.Figure:
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

def display_causal_graph(graph_data: Dict[str, Any]) -> None:
    """
    Display a causal graph visualization in Streamlit
    
    Args:
        graph_data: Dictionary containing nodes and edges of the causal graph
    """
    st.header("Causal Graph Visualization")
    
    if graph_data is not None:
        # Visualize the causal graph
        fig = create_causal_graph_visualization(graph_data)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No causal graph available. Upload a dataset and discover causal structure to visualize.")

def display_causal_effects(effects_data: Dict[str, Any]) -> None:
    """
    Display causal effects results in Streamlit
    
    Args:
        effects_data: Dictionary containing causal effects information
    """
    st.header("Causal Effects Results")
    
    if effects_data is not None:
        # Create a table of effects
        st.subheader("Average Treatment Effects")
        
        # Display the effects table
        st.table(effects_data['ate_table'])
        
        # Display any additional visualizations
        if 'visualization' in effects_data:
            st.subheader("Effect Visualization")
            st.plotly_chart(effects_data['visualization'], use_container_width=True)
    else:
        st.info("No causal effects available. Select treatment and outcome variables and estimate effects.")
