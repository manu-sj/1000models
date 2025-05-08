# 1000models - Extreme Model Parameterization Demo

A demonstration of how to train, manage, and serve thousands of specialized ML models from a single codebase.

## Extreme Parameterization

This project showcases how to:

- **Train thousands of models** with a single command
- Create a **separate model for each item×location combination**
- Automatically **select the best algorithm** for each combination
- **Scale effortlessly** from hundreds to thousands of models
- Maintain a **single entry point** while handling massive complexity underneath

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Process demand data and upload to feature store
python feature_pipeline.py

# Train 1000+ individual models (one per item-location)
python training_pipeline.py

# Generate forecasts using all trained models
python inference_pipeline.py
```

## How It Works

The system operates through three streamlined pipelines:

1. **Feature Pipeline**: 
   - Processes demand quantity data by item and location
   - Uploads data to Hopsworks feature store
   - Creates a central feature group with standardized format
   - Ensures data consistency and accessibility

2. **Training Pipeline**: 
   - Creates a feature view from the feature group
   - Identifies all unique item×location combinations
   - Trains both RandomForest and XGBoost models for each combination
   - Selects the better model based on RMSE performance
   - Handles insufficient data gracefully with minimum threshold checks
   - Stores models in Hopsworks model registry with metadata
   - Generates a comprehensive model performance report

3. **Inference Pipeline**: 
   - Retrieves the best model for each item-location combination
   - Generates forecasts for future time periods
   - Creates visualizations of predicted demand
   - Outputs forecasts in structured format
   - Optionally uploads forecasts back to the feature store

## Configuring Scale

Control the scale with simple parameters:

```bash
# Train models for all items in a specific location
python training_pipeline.py --location 3

# Generate forecasts for a specific item across all locations
python inference_pipeline.py --item 9684698
```

## Performance at Scale

The solution automatically:
- Parallelizes where possible
- Tracks performance across the entire model fleet
- Maintains summary statistics for all models
- Generates visualizations to understand model behavior

## Production Readiness

Built on enterprise-grade Hopsworks for:
- Feature storage and versioning
- Model registry and governance
- Online model serving
- Monitoring and tracking