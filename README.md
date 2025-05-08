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

## Inference Pipeline

The system uses a sophisticated batch inference approach that leverages Hopsworks feature views to ensure consistent transformations between training and prediction.

### Running Batch Inference

```bash
# Basic inference for a specific item and location 
python inference_pipeline.py --item 9684698 --location 3

# Generate forecasts for all items and locations
python inference_pipeline.py

# Forecast 6 months starting from July 2023
python inference_pipeline.py --start-year 2023 --start-month 7 --periods 6

# Forecast full year 2024 for a specific item
python inference_pipeline.py --item 8204334 --start-year 2024 --start-month 1 --periods 12

# Generate forecasts for all items in location 3 for Q1 2023
python inference_pipeline.py --location 3 --start-year 2023 --start-month 1 --periods 3

# Use a different feature group (if you've created a custom one)
python inference_pipeline.py --feature-group custom_demand_features
```

### Advanced Features

1. **Feature Transformations**: The pipeline automatically applies the same transformations used during training through the feature view, ensuring model input consistency.

2. **Best Model Selection**: For each item-location, the system:
   - Retrieves the model with the lowest RMSE 
   - Falls back to the latest version if best model can't be determined
   - Shows which model version is being used in logs

3. **Robust Fallbacks**: If feature view transformations encounter issues, the system uses a seasonal pattern approach as a fallback, ensuring predictions are always generated.

### Inference Outputs

The inference pipeline generates several outputs:

1. **CSV Files** (in the `forecasts/` directory):
   - `item_demand_forecast.csv`: Detailed forecast for each item-location-period
   - `location_demand_summary.csv`: Aggregated demand by location and time period

2. **Visualizations** (in the `forecasts/plots/` directory):
   - Individual forecast charts for each item-location
   - Heat map of total demand by location and time period

3. **Feature Store** (optional):
   - Forecasts are uploaded to a feature group named "demand_forecast"
   - Can be used for downstream applications or dashboards

### Technical Details

The inference uses `feature_view.get_batch_data(transformed=True, write=False)` to apply the same transformations used during training, ensuring feature name consistency without writing to the feature store.

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