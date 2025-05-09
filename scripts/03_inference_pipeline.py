import hopsworks
import pandas as pd
import argparse
import os
import joblib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def predict(project_name='models1000', model_name='demand_forecaster', 
           item_id=None, location_id=3, time_bucket=None):
    """
    Simple prediction function for a specific item and location
    """
    # Connect to Hopsworks
    project = hopsworks.login(
        host=os.getenv("HOST"),
        port=os.getenv("PORT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=project_name or os.getenv("PROJECT")
    )
    
    # Get model registry
    mr = project.get_model_registry()
    
    # Format time bucket if not provided
    if not time_bucket:
        now = datetime.now()
        time_bucket = int(f"{now.year}{now.month:02d}")
    
    # Get model for this item-location
    model_prefix = f"{model_name}_item{item_id}_loc{location_id}"
    
    try:
        # Get the best model
        model_instance = mr.get_best_model(
            name=model_prefix, 
            evaluation_metric="rmse", 
            sort_metrics_by="min"
        )
        
        # Download and load model
        model_dir = model_instance.download()
        if os.path.exists(os.path.join(model_dir, "model.joblib")):
            model = joblib.load(os.path.join(model_dir, "model.joblib"))
        else:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(model_dir, "model.json"))
        
        # Make prediction
        inference_data = pd.DataFrame([{'time_bucket': time_bucket}])
        prediction = float(model.predict(inference_data)[0])
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Return result
        return {
            "item_id": item_id,
            "location_id": location_id,
            "time_period": time_bucket,
            "predicted_demand": prediction,
            "metrics": model_instance.get_metrics()
        }
            
    except Exception:
        return None

def batch_predict(project_name, model_name, items, locations, time_period):
    """
    Batch prediction for multiple item-location combinations
    """
    results = []
    
    for item in items:
        for loc in locations:
            prediction = predict(
                project_name=project_name,
                model_name=model_name,
                item_id=item,
                location_id=loc,
                time_bucket=time_period
            )
            
            if prediction:
                results.append(prediction)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Pipeline')
    parser.add_argument('--project', type=str, help='Hopsworks project name')
    parser.add_argument('--model-name', type=str, default='demand_forecaster',
                      help='Base name for the models')
    parser.add_argument('--item', type=int, required=True, 
                      help='Item ID to predict demand for')
    parser.add_argument('--location', type=int, default=3,
                      help='Location ID (defaults to 3)')
    parser.add_argument('--time-period', type=int,
                      help='Time period in YYYYMM format')
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
        
        # Run batch prediction
        results = batch_predict(
            project_name=args.project,
            model_name=args.model_name,
            items=items,
            locations=locations,
            time_period=args.time_period
        )
        
        # Display results
        if len(results) > 0:
            print("\nBatch Prediction Results:")
            print(results)
    else:
        # Single prediction
        result = predict(
            project_name=args.project,
            model_name=args.model_name,
            item_id=args.item,
            location_id=args.location,
            time_bucket=args.time_period
        )
        
        if result:
            print(f"\nDemand Prediction Result:")
            print(f"Item: {result['item_id']}")
            print(f"Location: {result['location_id']}")
            print(f"Time Period: {result['time_period']}")
            print(f"Predicted Demand: {result['predicted_demand']:.2f} units")
            
            print(f"\nModel Performance Metrics:")
            for metric, value in result['metrics'].items():
                print(f"{metric}: {value:.4f}")
        else:
            print("No prediction available for the specified item-location combination")