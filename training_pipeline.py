import hopsworks
import joblib
import pandas as pd
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def main(project_name='models1000', feature_group_name='demand_features', version=1,
         model_name='demand_forecaster', model_version=1, test_size=0.2, location_id=None):
    """
    Training pipeline to build demand forecasting models per item-location
    """
    print("🔮 Connecting to Hopsworks")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOST")
    port = os.getenv("PORT")
    
    if not api_key:
        print("⚠️ No API key found in environment variables")
        return
        
    if project_name:
        project = hopsworks.login(host=host, port=port, api_key_value=api_key, project=project_name)
    else:
        project_name = os.getenv("PROJECT")
        project = hopsworks.login(host=host, port=port, api_key_value=api_key, project=project_name)
    
    fs = project.get_feature_store()
    
    print(f"🪄 Retrieving Feature Group: {feature_group_name}")
    demand_fg = fs.get_feature_group(
        name=feature_group_name,
        version=version,
    )
    
    print("🕵🏻‍♂️ Feature Group Investigation")
    print(demand_fg.show(5))
    
    print("💼 Feature Selection")
    # Define query with proper feature selection
    query = demand_fg.select_all()
    
    # Apply filters if needed
    if location_id is not None:
        query = query.filter(demand_fg.loc_id == location_id)
    
    print("🤖 Setting up transformation functions")
    from hopsworks.hsfs.builtin_transformations import standard_scaler
    
    # Get sample data to identify categorical columns
    sample_data = query.show(5)
    # Identify categorical columns - assume string/object types are categorical
    cat_cols = sample_data.select_dtypes(include=['object']).columns.tolist()
    
    # Define transformations - standard scaling for numeric features
    transformation_functions = []
    numeric_cols = [col for col in sample_data.columns if col not in cat_cols + ['datetime', 'repetitive_demand_quantity']]
    for col in numeric_cols:
        transformation_functions.append(standard_scaler(col))
        
    print("⚙️ Creating Feature View")
    # Get feature view or create one with transformations
    try:
        feature_view = fs.get_feature_view(name=f"{feature_group_name}_view", version=version)
    except:
        feature_view = fs.create_feature_view(
            name=f"{feature_group_name}_view",
            version=version,
            description="Feature view for demand forecasting",
            labels=["repetitive_demand_quantity"],
            transformation_functions=transformation_functions,
            query=query
        )
    
    print("🏋️ Creating Training Dataset")
    # Use train_test_split from feature view for proper splitting
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_size=test_size)
    
    # Get unique items and locations
    items = X_train['sp_id'].unique()
    locations = X_train['loc_id'].unique() if location_id is None else [location_id]
    
    # Calculate total number of models
    total_models = len(items) * len(locations)
    
    # Get model registry
    mr = project.get_model_registry()
    
    print(f"🧬 Training {total_models} models (items: {len(items)} × locations: {len(locations)})")
    
    # Counter for progress tracking
    model_counter = 0
    
    # Dictionary to store metrics for all models
    all_model_metrics = {}
    
    # Loop through each item and location combination
    for item in items:
        for loc in locations:
            model_counter += 1
            if model_counter % 10 == 0 or model_counter == 1:
                print(f"Training model {model_counter}/{total_models} (Item: {item}, Location: {loc})")
            
            # Filter data for this item-location combination
            item_loc_mask_train = (X_train['sp_id'] == item) & (X_train['loc_id'] == loc)
            item_loc_mask_test = (X_test['sp_id'] == item) & (X_test['loc_id'] == loc)
            
            # Skip if we don't have enough data for this combination
            if sum(item_loc_mask_train) < 10 or sum(item_loc_mask_test) < 5:
                print(f"⚠️ Skipping Item: {item}, Location: {loc} due to insufficient data")
                continue
            
            X_train_item = X_train[item_loc_mask_train].drop(['sp_id', 'loc_id', 'datetime'], axis=1)
            y_train_item = y_train[item_loc_mask_train]
            X_test_item = X_test[item_loc_mask_test].drop(['sp_id', 'loc_id', 'datetime'], axis=1)
            y_test_item = y_test[item_loc_mask_test]
        
            # Feature engineering - convert time_bucket to year and month
            for df in [X_train_item, X_test_item]:
                if not df.empty:
                    df['year'] = df['time_bucket'].astype(str).str[:4].astype(int)
                    df['month'] = df['time_bucket'].astype(str).str[4:].astype(int)
                    df.drop('time_bucket', axis=1, inplace=True)
            
            # Skip if dataframes are empty after processing
            if X_train_item.empty or X_test_item.empty:
                print(f"⚠️ Skipping Item: {item}, Location: {loc} due to empty data after processing")
                continue
            
            # Train both models for comparison
            model_prefix = f"{model_name}_item{item}_loc{loc}"
            
            try:
                # Train RandomForest
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(X_train_item, y_train_item)
                
                # Train XGBoost
                xgb_model = XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X_train_item, y_train_item)
                
                # Evaluate models
                models = {
                    "RandomForest": rf_model,
                    "XGBoost": xgb_model
                }
                
                best_model = None
                best_rmse = float('inf')
                best_metrics = {}
                
                for model_type, model in models.items():
                    # Make predictions
                    y_pred = model.predict(X_test_item)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(y_test_item, y_pred)
                    rmse = mean_squared_error(y_test_item, y_pred, squared=False)
                    r2 = r2_score(y_test_item, y_pred)
                    
                    metrics = {
                        "mae": mae,
                        "rmse": rmse,
                        "r2": r2
                    }
                    
                    if model_counter % 10 == 0 or model_counter == 1:
                        print(f"  {model_type} Metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
                    
                    # Track best model
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model
                        best_model_type = model_type
                        best_metrics = metrics
                
                # Store metrics for this item-location combination
                all_model_metrics[f"item_{item}_loc_{loc}"] = {
                    "model_type": best_model_type,
                    "metrics": best_metrics
                }
        
                # Create model directory
                model_dir = model_prefix
                images_dir = os.path.join(model_dir, "images")
                os.makedirs(images_dir, exist_ok=True)
                
                # Create feature importance plot
                plt.figure(figsize=(10, 6))
                if best_model_type == "RandomForest":
                    importances = best_model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    features = X_train_item.columns
                    
                    plt.title(f'Feature Importances for Item {item}, Location {loc}')
                    plt.bar(range(X_train_item.shape[1]), importances[indices], align='center')
                    plt.xticks(range(X_train_item.shape[1]), [features[i] for i in indices], rotation=90)
                    plt.tight_layout()
                else:  # XGBoost
                    xgb_model.get_booster().feature_names = list(X_train_item.columns)
                    xgb_model.get_booster().feature_importances = list(X_train_item.columns)
                    xgb_model.plot_importance(max_num_features=10)
                    plt.title(f'Feature Importances for Item {item}, Location {loc}')
                    plt.tight_layout()
                
                plt.savefig(os.path.join(images_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Save model
                if best_model_type == "RandomForest":
                    joblib.dump(best_model, os.path.join(model_dir, "model.joblib"))
                else:  # XGBoost
                    best_model.save_model(os.path.join(model_dir, "model.json"))
                
                # Register model in Hopsworks
                model_api = mr.python.create_model(
                    name=model_prefix,
                    metrics=best_metrics,
                    version=model_version,
                    description=f"Demand forecasting model for item {item}, location {loc} using {best_model_type}",
                    input_example=X_train_item.iloc[0].to_dict(),
                    feature_view=feature_view
                )
                
                # Upload the model and artifacts
                model_api.save(model_dir)
                
                if model_counter % 10 == 0 or model_counter == 1:
                    print(f"  ✅ Saved model for item {item}, location {loc} using {best_model_type}")
                
            except Exception as e:
                print(f"⚠️ Error training model for Item: {item}, Location: {loc}: {str(e)}")
                continue
    
    # Save overall metrics summary
    metrics_dir = "model_metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics to JSON
    with open(os.path.join(metrics_dir, "all_model_metrics.json"), 'w') as f:
        json.dump(all_model_metrics, f, indent=2)
    
    # Create summary statistics
    model_types = [data["model_type"] for data in all_model_metrics.values()]
    rf_count = model_types.count("RandomForest")
    xgb_count = model_types.count("XGBoost")
    
    # Get mean metrics across all models
    mae_values = [data["metrics"]["mae"] for data in all_model_metrics.values()]
    rmse_values = [data["metrics"]["rmse"] for data in all_model_metrics.values()]
    r2_values = [data["metrics"]["r2"] for data in all_model_metrics.values()]
    
    # Create summary report
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
    
    print(f"✅ Training pipeline completed successfully: {len(all_model_metrics)} models trained")
    print(f"   RandomForest models: {rf_count}, XGBoost models: {xgb_count}")
    print(f"   Average RMSE: {summary['average_metrics']['rmse']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Pipeline Parameters')
    parser.add_argument('--project', type=str, help='Hopsworks project name')
    parser.add_argument('--feature-group', type=str, default='demand_features', 
                        help='Feature group name')
    parser.add_argument('--version', type=int, default=1, 
                        help='Feature group version')
    parser.add_argument('--model-name', type=str, default='demand_forecaster',
                        help='Base name for the models')
    parser.add_argument('--model-version', type=int, default=1,
                        help='Model version')
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
        model_version=args.model_version,
        test_size=args.test_size,
        location_id=args.location
    )