import hopsworks
import joblib
import pandas as pd
import numpy as np
import argparse
import os
import json
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def get_feature_view(fs, feature_group_name, version):
    """Create feature view for training"""
    # Get feature group
    demand_fg = fs.get_feature_group(
        name=feature_group_name,
        version=version
    )
    
    # Define query with all features
    query = demand_fg.select_all()
    
    # Apply transformations
    from hopsworks.hsfs.builtin_transformations import label_encoder
    transformation_functions = [label_encoder("loc_id")]
    
    # Create feature view
    feature_view = fs.get_or_create_feature_view(
        name=f"{feature_group_name}_view",
        version=version,
        description="Feature view for demand forecasting",
        labels=["repetitive_demand_quantity"],
        query=query,
        transformation_functions=transformation_functions
    )
    
    return feature_view, query

def train_model(item, loc, feature_view, test_size=0.2):
    """Train model for specific item and location"""
    # Filter for this item-location
    item_loc_fv = feature_view.filter(
        (feature_view.get_feature('sp_id') == item) & 
        (feature_view.get_feature('loc_id') == loc)
    )
    
    # Split data
    X_train, X_test, y_train, y_test = item_loc_fv.train_test_split(test_size=test_size)
    
    # Skip if not enough data
    if len(X_train) < 10 or len(X_test) < 5:
        return None
    
    # Remove ID columns
    X_train = X_train.drop(['sp_id', 'loc_id', 'datetime'], axis=1, errors='ignore')
    X_test = X_test.drop(['sp_id', 'loc_id', 'datetime'], axis=1, errors='ignore')
    
    # Train models
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    for model_type, model in models.items():
        model.fit(X_train, y_train)
    
    # Evaluate and find best model
    best_model = None
    best_rmse = float('inf')
    best_metrics = {}
    best_model_type = None
    
    for model_type, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
        
        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_type = model_type
            best_metrics = metrics
    
    return {
        "model": best_model,
        "model_type": best_model_type,
        "metrics": best_metrics,
        "train_example": X_train.iloc[0].to_dict() if not X_train.empty else None
    }

def save_model(item, loc, model_result, feature_view, model_registry, model_name):
    """Save and register model with Hopsworks"""
    model = model_result["model"]
    model_type = model_result["model_type"]
    metrics = model_result["metrics"]
    train_example = model_result["train_example"]
    
    # Create model directory
    model_prefix = f"{model_name}_item{item}_loc{loc}"
    model_dir = model_prefix
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    if model_type == "RandomForest":
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    else:  # XGBoost
        model.save_model(os.path.join(model_dir, "model.json"))
    
    # Register model with Hopsworks
    model_api = model_registry.python.create_model(
        name=model_prefix,
        metrics=metrics,
        description=f"Demand forecaster for item {item}, location {loc}",
        input_example=train_example,
        feature_view=feature_view
    )
    
    # Upload the model
    model_api.save(model_dir)
    
    # Clean up local model dir
    import shutil
    shutil.rmtree(model_dir, ignore_errors=True)
    
    return metrics

def main(project_name='models1000', feature_group_name='demand_features', version=1,
         model_name='demand_forecaster', test_size=0.2, location_id=None):
    """Main training function"""
    # Connect to Hopsworks
    project = hopsworks.login(
        host=os.getenv("HOST"),
        port=os.getenv("PORT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=project_name or os.getenv("PROJECT")
    )
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # Create feature view
    feature_view, query = get_feature_view(fs, feature_group_name, version)
    
    # Get unique items and locations
    items = query.read(limit=1000)['sp_id'].unique()
    locations = query.read(limit=1000)['loc_id'].unique() if location_id is None else [location_id]
    
    # Track metrics and progress
    total_models = len(items) * len(locations)
    print(f"Training {total_models} models (items: {len(items)} × locations: {len(locations)})")
    
    all_model_metrics = {}
    model_counter = 0
    
    # Train models for each item-location
    for item in items:
        for loc in locations:
            model_counter += 1
            
            # Display progress occasionally
            if model_counter % 5 == 0 or model_counter == 1:
                print(f"Training model {model_counter}/{total_models} (Item: {item}, Location: {loc})")
            
            # Train model
            model_result = train_model(item, loc, feature_view, test_size)
            
            if model_result:
                # Save model
                metrics = save_model(item, loc, model_result, feature_view, mr, model_name)
                
                # Store metrics
                all_model_metrics[f"item_{item}_loc_{loc}"] = {
                    "model_type": model_result["model_type"],
                    "metrics": metrics
                }
                
                if model_counter % 5 == 0 or model_counter == 1:
                    print(f"  Model: {model_result['model_type']}, RMSE: {metrics['rmse']:.2f}")
    
    # Save metrics summary
    if all_model_metrics:
        metrics_dir = "model_metrics"
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Save detailed metrics
        with open(os.path.join(metrics_dir, "all_model_metrics.json"), 'w') as f:
            json.dump(all_model_metrics, f, indent=2)
        
        # Create summary stats
        model_types = [data["model_type"] for data in all_model_metrics.values()]
        rf_count = model_types.count("RandomForest")
        xgb_count = model_types.count("XGBoost")
        
        # Calculate average metrics
        mae_values = [data["metrics"]["mae"] for data in all_model_metrics.values()]
        rmse_values = [data["metrics"]["rmse"] for data in all_model_metrics.values()]
        r2_values = [data["metrics"]["r2"] for data in all_model_metrics.values()]
        
        summary = {
            "total_models_trained": len(all_model_metrics),
            "model_type_distribution": {
                "RandomForest": rf_count,
                "XGBoost": xgb_count
            },
            "average_metrics": {
                "mae": sum(mae_values) / len(mae_values) if mae_values else 0,
                "rmse": sum(rmse_values) / len(rmse_values) if rmse_values else 0,
                "r2": sum(r2_values) / len(r2_values) if r2_values else 0
            }
        }
        
        # Save summary
        with open(os.path.join(metrics_dir, "training_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training completed: {len(all_model_metrics)} models trained")
        print(f"RandomForest: {rf_count}, XGBoost: {xgb_count}")
        print(f"Average RMSE: {summary['average_metrics']['rmse']:.2f}")
    
    # All model directories are cleaned up during training

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline Parameters')
    parser.add_argument('--project', type=str, help='Hopsworks project name')
    parser.add_argument('--feature-group', type=str, default='demand_features', 
                        help='Feature group name')
    parser.add_argument('--version', type=int, default=1, 
                        help='Feature group version')
    parser.add_argument('--model-name', type=str, default='demand_forecaster',
                        help='Base name for the models')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--location', type=int, default=None,
                        help='Filter training for a specific location')
    
    args = parser.parse_args()
    
    main(
        project_name=args.project,
        feature_group_name=args.feature_group,
        version=args.version,
        model_name=args.model_name,
        test_size=args.test_size,
        location_id=args.location
    )