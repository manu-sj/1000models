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

# Generate a prediction for a specific item
python inference_pipeline.py --item 9684698
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
   - Retrieves the best model for a specific item-location combination
   - Applies the same feature transformations used during training
   - Generates a demand prediction for the requested time period
   - Displays model performance metrics alongside the prediction

## Configuring Scale

Control the scale with simple parameters:

```bash
# Train models for all items in a specific location
python training_pipeline.py --location 3

# Generate a prediction for a specific item
python inference_pipeline.py --item 9684698
```

## Inference Pipeline

The system uses Hopsworks feature views to ensure consistent transformations between training and prediction.

### Running Inference

```bash
# Basic inference for a specific item (location defaults to 3)
python inference_pipeline.py --item 9684698

# Specify a different location if needed
python inference_pipeline.py --item 9684698 --location 3

# Predict for a specific time period
python inference_pipeline.py --item 8204334 --start-year 2023 --start-month 7

# Use a different feature group (if you've created a custom one)
python inference_pipeline.py --item 9684698 --feature-group custom_demand_features
```

### Technical Details

The inference uses `feature_view.get_batch_data(transformed=True, write=False)` to apply the same transformations used during training, ensuring feature name consistency and accurate predictions.

## Performance at Scale

The solution automatically:
- Parallelizes training where possible
- Tracks performance across the entire model fleet
- Maintains summary statistics for all models
- Creates a separate model for each item-location combination

## Production Readiness

Built on enterprise-grade Hopsworks for:
- Feature storage and versioning
- Model registry and governance
- Online model serving
- Monitoring and tracking