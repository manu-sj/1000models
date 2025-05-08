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

1. **Feature Pipeline**: Processes data once and stores it centrally
2. **Training Pipeline**: Automatically:
   - Identifies all unique item×location combinations
   - Trains separate RF and XGBoost models for each
   - Selects the better model based on performance
   - Handles insufficient data gracefully
   - Generates a comprehensive model performance report
3. **Inference Pipeline**: Uses the best model for each combination to generate forecasts

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