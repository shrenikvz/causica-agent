# System Architecture: Web-based AI Agent for Causal Inference

## Overview

This document outlines the architecture for a web-based AI agent that integrates LangChain and Microsoft's Causica library to facilitate collaborative causal inference between a human engineer and an AI assistant. The system enables natural language interaction, dataset upload, causal discovery, and inference through an intuitive interface.

## System Components

The system consists of four main components:

1. **Web Interface**: A user-facing frontend for interaction and visualization
2. **LangChain Agent**: An AI agent powered by OpenAI's o3-mini model
3. **Causica Integration Layer**: A wrapper around the Causica library for causal inference
4. **Data Processing Layer**: Handles dataset management and transformation

### Component Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Web Interface  │◄────┤ LangChain Agent │◄────┤ Data Processing │
│  (Frontend)     │     │ (AI Controller) │     │     Layer       │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐             │
         └─────────────►│    Causica      │◄────────────┘
                        │ Integration Layer│
                        └─────────────────┘
```

## Detailed Component Descriptions

### 1. Web Interface

**Technology**: FastAPI + React (or Streamlit for rapid development)

**Responsibilities**:
- Provide a chat interface for natural language interaction
- Enable CSV dataset upload functionality
- Display causal graphs and inference results visually
- Show explanations and interpretations from the AI agent
- Allow user approval at key decision points

**Key Features**:
- Real-time chat with the AI agent
- File upload for CSV datasets
- Interactive causal graph visualization
- Results display with interpretations
- Confirmation dialogs for key decisions

### 2. LangChain Agent

**Technology**: LangChain + OpenAI API (o3-mini model)

**Responsibilities**:
- Interpret user intent from natural language
- Explain understanding before taking action
- Coordinate with Causica integration layer
- Provide explanations of causal inference results
- Handle error detection and recovery

**Key Components**:
- OpenAI API integration with o3-mini model
- LangChain agent with custom tools for Causica interaction
- Conversation memory for context retention
- Error handling and recovery mechanisms
- Explanation generation for causal inference results

### 3. Causica Integration Layer

**Technology**: Python wrapper around Causica library

**Responsibilities**:
- Interface with Causica's DECI model for causal discovery
- Perform causal inference operations
- Execute simulation-based causal analysis
- Handle model training and evaluation
- Provide results in a format suitable for visualization

**Key Features**:
- Causal discovery using DECI
- Treatment effect estimation
- Counterfactual analysis
- Model training and hyperparameter management
- Result formatting and interpretation

### 4. Data Processing Layer

**Technology**: Python with pandas/numpy

**Responsibilities**:
- Process uploaded CSV datasets
- Transform data into Causica-compatible format (TensorDict)
- Handle missing values and data preprocessing
- Manage dataset versioning and storage
- Provide data statistics and summaries

**Key Features**:
- CSV parsing and validation
- Data transformation to TensorDict format
- Missing value handling
- Data normalization and preprocessing
- Dataset metadata management

## Data Flow

1. **User Input Flow**:
   - User uploads CSV dataset or sends natural language query
   - Web interface forwards data to LangChain agent
   - Agent interprets intent and determines required actions
   - Agent explains understanding and seeks approval if needed

2. **Causal Discovery Flow**:
   - Processed dataset is passed to Causica integration layer
   - DECI model performs causal discovery
   - Discovered causal graph is returned to agent
   - Agent explains results and sends visualization to web interface

3. **Causal Inference Flow**:
   - User specifies variables of interest for causal analysis
   - Agent formulates appropriate causal queries
   - Causica layer performs inference (e.g., treatment effect estimation)
   - Results are processed, explained, and visualized for the user

4. **Error Handling Flow**:
   - Errors during execution are caught by appropriate layer
   - Agent attempts to resolve errors automatically when possible
   - User is informed of errors and resolution steps
   - Alternative approaches are suggested when needed

## API Endpoints

### Web Interface API

1. `/chat` - WebSocket endpoint for real-time chat
2. `/upload` - POST endpoint for dataset upload
3. `/models` - GET endpoint to retrieve available causal models
4. `/discover` - POST endpoint to trigger causal discovery
5. `/infer` - POST endpoint to perform causal inference
6. `/visualize` - GET endpoint to retrieve visualizations

### LangChain Agent API

1. `interpret_message(message)` - Interpret user message
2. `explain_understanding(intent)` - Generate explanation of understanding
3. `execute_causica_operation(operation, params)` - Execute Causica operations
4. `generate_explanation(results)` - Generate explanation of results
5. `handle_error(error)` - Handle and explain errors

### Causica Integration API

1. `load_dataset(data)` - Load and prepare dataset
2. `discover_causal_structure(data, params)` - Perform causal discovery
3. `estimate_treatment_effect(model, treatment, outcome)` - Estimate causal effects
4. `perform_counterfactual(model, intervention)` - Perform counterfactual analysis
5. `visualize_graph(model)` - Generate graph visualization

## Technology Stack

- **Backend**: Python 3.10+, FastAPI
- **Frontend**: React or Streamlit
- **AI**: LangChain, OpenAI API (o3-mini model)
- **Causal Inference**: Microsoft Causica (DECI model)
- **Data Processing**: Pandas, NumPy, TensorDict
- **Visualization**: NetworkX, Plotly
- **Deployment**: Docker, Poetry for dependency management

## Security Considerations

- User data is processed locally and not stored permanently
- OpenAI API keys are managed securely through environment variables
- Input validation is performed on all user-provided data
- Error messages are sanitized to prevent information leakage

## Future Extensions

- Support for additional data formats beyond CSV
- Integration with more causal inference methods from Causica
- Enhanced visualization capabilities for complex causal structures
- Automated report generation for causal analysis results
- Collaborative features for multiple users working on the same dataset
