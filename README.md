# 1000models - Extreme Model Parameterization

A demonstration of how parameterization enables scaling from a single data source to thousands of specialized production-grade ML models.

```
┌───────────┐     ┌──────────────┐     ┌───────────────────────────┐
│ Data      │     │ Feature      │     │       Model Training      │
│ Source    │────▶│ Store        │────▶│                           │
│           │     │              │     │  ┌─────┐  ┌─────┐  ┌─────┐│
└───────────┘     └──────────────┘     │  │Model│  │Model│  │Model││
                                       │  │ 1   │  │ 2   │  │ ... ││
                                       │  └─────┘  └─────┘  └─────┘│
                                       └───────────────┬───────────┘
                                                       │
                                                       ▼
┌────────────────┐     ┌──────────────┐     ┌─────────────────────┐
│ Production     │     │ Best Model   │     │   Model Registry    │
│ Predictions    │◀────│ Selection    │◀────│                     │
│                │     │              │     │ (1000+ models with  │
└────────────────┘     └──────────────┘     │  performance data)  │
                                            └─────────────────────┘
```

## Parameterization Strategy

1000models demonstrates how to use parameterization to:

- Transform a single data source into **thousands of specialized ML models**
- Train a **separate model for each item×location combination**
- Automatically **select the best algorithm** for each scenario
- **Scale effortlessly** from hundreds to thousands of models
- Maintain a **single MLOps workflow** while handling massive complexity

## Interactive Notebooks

The primary way to use this project is through our Jupyter notebooks:

- **[Feature Pipeline](notebooks/feature_pipeline.ipynb)**: Prepare and upload demand data
- **[Training Pipeline](notebooks/training_pipeline.ipynb)**: Train specialized models by parameter
- **[Inference Pipeline](notebooks/inference_pipeline.ipynb)**: Generate predictions with best model selection

Each notebook is self-contained and includes detailed guidance.

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/1000models.git
cd 1000models

# Install dependencies
pip install -r requirements.txt

# Test connection to Hopsworks
python test_connection.py
```

## Parameterization Workflow

The system follows this parameterized approach:

1. **Feature Preparation**
   - Process data into a single feature store
   - Create a unified view of demand data by item and location

2. **Parameterized Model Training**
   - Automatically generate training subsets for each parameter combination
   - Train competing models (RandomForest and XGBoost) for each parameter set
   - Select best model based on performance metrics
   - Register all models with full parameter metadata

3. **Intelligent Model Selection**
   - Dynamically select the best-performing model for each parameter combination
   - Generate predictions using the optimal model for each use case

## Design Principles

- **Parameter-Driven Architecture**: Structure determined by business parameters
- **Automatic Model Selection**: System chooses best model type for each parameter combo
- **Notebook-First Interface**: Interactive, self-documenting workflow
- **MLOps Integration**: Complete cycle from data to production predictions

## Command-Line Interface

For automated workflows, script versions of each pipeline are also available:

```bash
# Process demand data
python scripts/feature_pipeline.py

# Train models (all items, all locations)
python scripts/training_pipeline.py

# Train models for a specific parameter set
python scripts/training_pipeline.py --location 3

# Generate a prediction using the best model for these parameters
python scripts/inference_pipeline.py --item 9684698
```

## Technical Stack

- **Feature Management**: Hopsworks Feature Store
- **Model Training**: Scikit-learn and XGBoost with automatic selection
- **Model Registry**: Hopsworks Model Registry with parameter metadata
- **Visualization**: Matplotlib (in notebooks)

---

This project demonstrates how extreme parameterization creates specialized models that significantly improve forecasting accuracy while maintaining a clean, unified MLOps workflow.