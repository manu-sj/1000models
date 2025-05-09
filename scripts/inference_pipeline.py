import hopsworks
import pandas as pd
import argparse
import os
import joblib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main(project_name='models1000', model_name='demand_forecaster', feature_group_name='demand_features',
         item_id=None, location_id=3, start_year=2021, start_month=1, periods=1):
    """
    Simplified inference pipeline to predict demand for a specific item and location
    """
    print("Connecting to Hopsworks")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOST")
    port = os.getenv("PORT")

    project = hopsworks.login(host=host, port=port, api_key_value=api_key, project=project_name or os.getenv("PROJECT"))
    
    # Get model registry
    mr = project.get_model_registry()
    
    # Validate parameters
    if item_id is None:
        print("Error: Please provide an item_id")
        return
    
    # Format time bucket for prediction period
    time_bucket = int(f"{start_year}{start_month:02d}")
    print(f"Predicting demand for item {item_id}, location {location_id}, time period {time_bucket}")
    
    # Look for the model with this item-location combination
    model_prefix = f"{model_name}_item{item_id}_loc{location_id}"
    
    try:
        # Get the best model based on RMSE
        model_instance = mr.get_best_model(name=model_prefix, 
                                        evaluation_metric="rmse", 
                                        sort_metrics_by="min")
        print(f"Found best model (version {model_instance.version}) for item {item_id}, location {location_id}")
        
        # Download model
        model_dir = model_instance.download()
        
        # Create inference data with just the time bucket
        inference_data = pd.DataFrame([{
            'time_bucket': time_bucket
        }])
        
        # Load the model (supports both joblib and XGBoost formats)
        if os.path.exists(os.path.join(model_dir, "model.joblib")):
            model = joblib.load(os.path.join(model_dir, "model.joblib"))
        else:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(model_dir, "model.json"))
        
        # Make prediction
        prediction = float(model.predict(inference_data)[0])
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Print the result
        print(f"\nDemand Prediction Results:")
        print(f"Item: {item_id}")
        print(f"Location: {location_id}")
        print(f"Time Period: {start_year}-{start_month:02d}")
        print(f"Predicted Demand: {prediction:.2f} units")
        
        # Get model metrics
        metrics = model_instance.get_metrics()
        print(f"\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return prediction
            
    except Exception:
        return None

def batch_predict(project_name, model_name, items, locations, time_period):
    """
    Batch prediction for multiple item-location combinations
    
    Args:
        project_name: Hopsworks project name
        model_name: Base model name 
        items: List of item IDs
        locations: List of location IDs
        time_period: Time period in YYYYMM format
        
    Returns:
        DataFrame with predictions for all item-location combinations
    """
    # Connect to Hopsworks
    api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    
    project = hopsworks.login(host=host, port=port, api_key_value=api_key, project=project_name or os.getenv("PROJECT"))
    mr = project.get_model_registry()
    
    results = []
    
    for item in items:
        for loc in locations:
            # Get model name for this item-location
            model_prefix = f"{model_name}_item{item}_loc{loc}"
            
            try:
                # Get the best model
                model_instance = mr.get_best_model(name=model_prefix, 
                                                evaluation_metric="rmse", 
                                                sort_metrics_by="min")
                
                # Download model
                model_dir = model_instance.download()
                
                # Load the model
                if os.path.exists(os.path.join(model_dir, "model.joblib")):
                    model = joblib.load(os.path.join(model_dir, "model.joblib"))
                else:
                    import xgboost as xgb
                    model = xgb.XGBRegressor()
                    model.load_model(os.path.join(model_dir, "model.json"))
                
                # Create inference data
                inference_data = pd.DataFrame([{
                    'time_bucket': time_period
                }])
                
                # Make prediction
                prediction = float(model.predict(inference_data)[0])
                prediction = max(0, prediction)  # Ensure non-negative
                
                # Get metrics
                metrics = model_instance.get_metrics()
                
                # Store result
                results.append({
                    'item_id': item,
                    'location_id': loc,
                    'time_period': time_period,
                    'predicted_demand': prediction,
                    'rmse': metrics.get('rmse', None),
                    'model_type': 'RandomForest' if os.path.exists(os.path.join(model_dir, "model.joblib")) else 'XGBoost'
                })
                
            except Exception:
                pass
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified Inference Pipeline')
    parser.add_argument('--project', type=str, help='Hopsworks project name')
    parser.add_argument('--model-name', type=str, default='demand_forecaster',
                      help='Base name for the models')
    parser.add_argument('--feature-group', type=str, default='demand_features',
                      help='Feature group name')
    parser.add_argument('--item', type=int, required=True, 
                      help='Item ID to predict demand for')
    parser.add_argument('--location', type=int, default=3,
                      help='Location ID (defaults to 3)')
    parser.add_argument('--start-year', type=int, default=2021,
                      help='Year for prediction')
    parser.add_argument('--start-month', type=int, default=1,
                      help='Month for prediction')
    parser.add_argument('--periods', type=int, default=1,
                      help='Number of periods to forecast (currently only 1 supported)')
    parser.add_argument('--batch', action='store_true', 
                      help='Perform batch prediction for multiple items or locations')
    parser.add_argument('--items', type=str, 
                      help='Comma-separated list of item IDs for batch prediction')
    parser.add_argument('--locations', type=str, 
                      help='Comma-separated list of location IDs for batch prediction')
    
    args = parser.parse_args()
    
    if args.batch:
        # Parse items and locations for batch prediction
        items = [int(x) for x in args.items.split(',')] if args.items else [args.item]
        locations = [int(x) for x in args.locations.split(',')] if args.locations else [args.location]
        
        # Format time period
        time_period = int(f"{args.start_year}{args.start_month:02d}")
        
        # Run batch prediction
        results = batch_predict(
            project_name=args.project,
            model_name=args.model_name,
            items=items,
            locations=locations,
            time_period=time_period
        )
        
        # Display results
        if len(results) > 0:
            print("\nBatch Prediction Results:")
            print(results)
    else:
        # Single prediction
        main(
            project_name=args.project,
            model_name=args.model_name,
            feature_group_name=args.feature_group,
            item_id=args.item,
            location_id=args.location,
            start_year=args.start_year,
            start_month=args.start_month,
            periods=args.periods
        )