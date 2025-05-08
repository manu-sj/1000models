# 1000models - Demand Forecasting Project

A scalable demand forecasting system using Hopsworks feature store that can create and manage hundreds of models.

## Project Structure

This project consists of three main pipelines:

1. **Feature Pipeline**: Processes raw demand data and uploads it to the Hopsworks feature store
2. **Training Pipeline**: Trains individual forecasting models for each item/location combination
3. **Inference Pipeline**: Generates forecasts using the trained models and uploads them to the feature store

## Prerequisites

- Python 3.8+
- Hopsworks account and API key

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with your Hopsworks credentials:

```
HOST=demo.hops.works
PORT=443
PROJECT=models1000
HOPSWORKS_API_KEY=your_api_key_here
```

## Usage

### Feature Pipeline

The feature pipeline loads the demand data and uploads it to the Hopsworks feature store.

```bash
python feature_pipeline.py [--project PROJECT_NAME] [--feature-group FEATURE_GROUP_NAME] [--version VERSION]
```

### Training Pipeline

The training pipeline creates and trains demand forecasting models for each item/location combination, comparing RandomForest and XGBoost models and selecting the best one.

```bash
python training_pipeline.py [--project PROJECT_NAME] [--feature-group FEATURE_GROUP_NAME] [--version VERSION] [--model-name MODEL_NAME] [--model-version MODEL_VERSION] [--test-size TEST_SIZE] [--location LOCATION_ID]
```

### Inference Pipeline

The inference pipeline generates forecasts using the trained models and visualizes the results.

```bash
python inference_pipeline.py [--project PROJECT_NAME] [--model-name MODEL_NAME] [--item ITEM_ID] [--location LOCATION_ID] [--start-year START_YEAR] [--start-month START_MONTH] [--periods PERIODS]
```

## Hopsworks Integration

This project uses Hopsworks for:
- Feature storage and versioning
- Feature transformations
- Feature views for model training
- Model registry for tracking models
- Online serving of models

## Best Practices

The pipelines follow these best practices:
- Proper feature engineering and selection
- Data transformation (scaling, encoding)
- Model comparison and selection
- Visualization of results
- Comprehensive logging
- Parameterized execution
- Version control of features and models