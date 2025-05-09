import hopsworks
import pandas as pd
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main(project_name='models1000', model_name='demand_forecaster', feature_group_name='demand_features',
         item_id=None, location_id=3, start_year=2021, start_month=1, periods=1):
    """
    Simplified inference pipeline to predict demand for a specific item and location
    """
    print("🔮 Connecting to Hopsworks")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    
    if not api_key:
        print("⚠️ No API key found in environment variables")
        return
        
    project = hopsworks.login(host=host, port=port, api_key_value=api_key, project=project_name or os.getenv("PROJECT"))
    
    # Get model registry and feature store
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
    # Validate parameters
    if item_id is None:
        print("⚠️ Error: Please provide an item_id")
        return
    
    # Format time bucket for prediction period
    time_bucket = int(f"{start_year}{start_month:02d}")
    print(f"🔍 Predicting demand for item {item_id}, location {location_id}, time period {time_bucket}")
    
    # Look for the model with this item-location combination
    model_prefix = f"{model_name}_item{item_id}_loc{location_id}"
    
    try:
        # Get the model from Hopsworks
        model_instance = mr.get_model(name=model_prefix, version=1)
        print(f"✅ Found model for item {item_id}, location {location_id}")
        
        # Download model
        model_dir = model_instance.download()
        
        # Get feature view for transformations
        fv_name = f"{feature_group_name}_view"
        feature_view = fs.get_feature_view(name=fv_name, version=1)
        
        # Initialize the feature view with a training dataset
        training_datasets = feature_view.get_training_datasets()
        if training_datasets and len(training_datasets) > 0:
            td_version = training_datasets[-1].version
            feature_view.init_batch_scoring(td_version)
        
        # Create inference data with the query parameters
        inference_data = pd.DataFrame([{
            'sp_id': item_id,
            'loc_id': location_id,
            'time_bucket': time_bucket,
            'datetime': datetime.now()
        }])
        
        # Apply transformations from feature view
        transformed_data = feature_view.get_batch_data(
            data=inference_data,
            transformed=True,
            write=False
        )
        
        # Remove the label column if present
        if 'repetitive_demand_quantity' in transformed_data.columns:
            transformed_data = transformed_data.drop('repetitive_demand_quantity', axis=1)
        
        # Load the model (supports both joblib and XGBoost formats)
        if os.path.exists(os.path.join(model_dir, "model.joblib")):
            import joblib
            model = joblib.load(os.path.join(model_dir, "model.joblib"))
        elif os.path.exists(os.path.join(model_dir, "model.json")):
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(model_dir, "model.json"))
        else:
            print(f"⚠️ Could not find model file in {model_dir}")
            return
        
        # Make prediction
        prediction = float(model.predict(transformed_data)[0])
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Print the result
        print(f"\n📊 Demand Prediction Results:")
        print(f"Item: {item_id}")
        print(f"Location: {location_id}")
        print(f"Time Period: {start_year}-{start_month:02d}")
        print(f"Predicted Demand: {prediction:.2f} units")
        
        # Get model metrics
        try:
            metrics = model_instance.get_metrics()
            print(f"\nModel Performance Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        except:
            pass
        
        return prediction
            
    except Exception as e:
        print(f"⚠️ Error during prediction: {str(e)}")
        print("Please ensure the model exists for this item-location combination.")
        print("You may need to run the training pipeline first.")
        return None

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
    
    args = parser.parse_args()
    
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