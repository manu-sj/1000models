import hopsworks
import pandas as pd
import numpy as np
import argparse
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

def main(project_name='models1000', model_name='demand_forecaster', item_id=None, location_id=None,
         start_year=2021, start_month=1, periods=12):
    """
    Inference pipeline to generate demand forecasts using trained models
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
    
    # Get model registry
    mr = project.get_model_registry()
    
    # Get feature store for metadata
    fs = project.get_feature_store()
    
    print("🔍 Getting item and location metadata")
    
    # Two approaches to get available items and locations:
    # 1. From feature store (preferred for complete list)
    # 2. From model registry (as fallback)
    
    available_items = []
    available_locations = []
    
    # Approach 1: Get metadata from feature store
    try:
        demand_fg = fs.get_feature_group("demand_features", version=1)
        # Get all unique items and locations if not specified
        query = demand_fg.select(['sp_id', 'loc_id']).distinct()
        df = query.read()
        
        available_items = df['sp_id'].unique().tolist()
        available_locations = df['loc_id'].unique().tolist()
        
        print(f"Found {len(available_items)} items and {len(available_locations)} locations in feature store")
    except Exception as e:
        print(f"⚠️ Could not retrieve metadata from feature store: {str(e)}")
    
    # Approach 2: If feature store approach failed or if we specified only location,
    # try to find models in model registry
    if not available_items or (not item_id and location_id):
        print("Checking available models in model registry...")
        try:
            # Get all models that match the pattern for this forecaster
            all_models = mr.get_models(name=f"{model_name}_item*")
            
            # Extract item and location IDs from model names
            for model in all_models:
                # Extract item ID from model name (format: "model_name_itemXXXX_locYY")
                model_name_parts = model.name.split('_')
                if len(model_name_parts) >= 3:
                    item_part = model_name_parts[2]
                    if item_part.startswith('item'):
                        try:
                            extracted_item = int(item_part[4:])
                            if extracted_item not in available_items:
                                available_items.append(extracted_item)
                        except:
                            pass
                
                # Extract location ID if present
                if len(model_name_parts) >= 4:
                    loc_part = model_name_parts[3]
                    if loc_part.startswith('loc'):
                        try:
                            extracted_loc = int(loc_part[3:])
                            if extracted_loc not in available_locations:
                                available_locations.append(extracted_loc)
                        except:
                            pass
            
            print(f"Found {len(available_items)} items and {len(available_locations)} locations from model registry")
        except Exception as e:
            print(f"⚠️ Could not retrieve models from model registry: {str(e)}")
    
    # Validate provided item_id and location_id if specified
    if item_id and available_items and item_id not in available_items:
        print(f"⚠️ Warning: Item {item_id} not found in available items")
        return
    
    if location_id and available_locations and location_id not in available_locations:
        print(f"⚠️ Warning: Location {location_id} not found in available locations")
        return
    
    # Filter the items and locations to forecast
    items_to_forecast = [item_id] if item_id else available_items
    locations_to_forecast = [location_id] if location_id else available_locations
    
    if not items_to_forecast or not locations_to_forecast:
        print("⚠️ No items or locations available for forecasting")
        return
    
    print(f"Will forecast for {len(items_to_forecast)} items and {len(locations_to_forecast)} locations")
    
    # Prepare for forecasting
    forecast_data = []
    model_metrics = {}
    
    print(f"🔮 Generating forecasts for {len(items_to_forecast)} items and {len(locations_to_forecast)} locations")
    
    # Create output directories
    output_dir = "forecasts"
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for loc in locations_to_forecast:
        for item in items_to_forecast:
            # Try to get item-location specific model first
            try:
                model_prefix = f"{model_name}_item{item}_loc{loc}"
                # Get the best model by RMSE (lowest is best)
                model_instance = mr.get_best_model(model_prefix, 'rmse', 'min')
                if model_instance:
                    print(f"Found best specific model for item {item} location {loc} (version {model_instance.version})")
                    model_type = "location_specific"
                else:
                    # If get_best_model returns None, try to get the latest version
                    models = mr.get_models(name=model_prefix)
                    if models and len(models) > 0:
                        # Get the latest version
                        model_instance = models[-1]
                        print(f"Found latest specific model for item {item} location {loc} (version {model_instance.version})")
                        model_type = "location_specific"
                    else:
                        raise Exception("No models found")
            except:
                # Try to get item model without location
                try:
                    model_prefix = f"{model_name}_item{item}"
                    # Get the best model by RMSE (lowest is best)
                    model_instance = mr.get_best_model(model_prefix, 'rmse', 'min')
                    if model_instance:
                        print(f"Found best general model for item {item} (version {model_instance.version})")
                        model_type = "item_specific"
                    else:
                        # If get_best_model returns None, try to get the latest version
                        models = mr.get_models(name=model_prefix)
                        if models and len(models) > 0:
                            # Get the latest version
                            model_instance = models[-1]
                            print(f"Found latest general model for item {item} (version {model_instance.version})")
                            model_type = "item_specific"
                        else:
                            raise Exception("No models found")
                except:
                    print(f"⚠️ No model found for item {item}")
                    continue
            
            # Download model
            print(f"Downloading model {model_prefix}")
            model_dir = model_instance.download()
            
            # Try to load model metrics
            try:
                metrics = model_instance.get_metrics()
                model_metrics[f"item_{item}_loc_{loc}"] = metrics
                print(f"Model metrics: {metrics}")
            except:
                print("Could not retrieve model metrics")
            
            # Determine model format and load accordingly
            if os.path.exists(os.path.join(model_dir, "model.joblib")):
                model = joblib.load(os.path.join(model_dir, "model.joblib"))
                model_format = "joblib"
            elif os.path.exists(os.path.join(model_dir, "model.json")):
                import xgboost as xgb
                model = xgb.XGBRegressor()
                model.load_model(os.path.join(model_dir, "model.json"))
                model_format = "xgboost"
            else:
                print(f"⚠️ Could not find model file in {model_dir}")
                continue
                
            print(f"Loaded {model_format} model for item {item}, location {loc}")
            
            # Generate forecast periods and create empty array for predictions
            time_buckets = []
            prediction_dates = []
            item_predictions = []
            
            for i in range(periods):
                year = start_year
                month = start_month + i
                
                # Handle month overflow
                while month > 12:
                    year += 1
                    month -= 12
                
                # Create time bucket
                time_bucket = int(f"{year}{month:02d}")
                time_buckets.append(time_bucket)
                prediction_dates.append(f"{year}-{month:02d}")
                
                # Prepare inference data
                inference_data = {
                    'year': year,
                    'month': month
                }
                
                # Get prediction
                prediction = float(model.predict(pd.DataFrame([inference_data]))[0])
                # Ensure prediction is non-negative
                prediction = max(0, prediction)
                item_predictions.append(prediction)
                
                # Add to forecast data
                forecast_data.append({
                    'sp_id': item,
                    'loc_id': loc,
                    'year': year,
                    'month': month,
                    'time_bucket': time_bucket,
                    'predicted_demand': prediction,
                    'model_type': model_type,
                    'forecast_date': datetime.now()
                })
            
            # Create forecast visualization
            plt.figure(figsize=(12, 6))
            plt.plot(prediction_dates, item_predictions, marker='o', linestyle='-', linewidth=2)
            plt.title(f'Demand Forecast for Item {item} at Location {loc}')
            plt.xlabel('Date')
            plt.ylabel('Predicted Demand')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join(plots_dir, f"forecast_item{item}_loc{loc}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create forecast dataframe
    if forecast_data:
        forecast_df = pd.DataFrame(forecast_data)
        
        # Create summary stats by location
        summary_by_loc = forecast_df.groupby(['loc_id', 'time_bucket']).agg(
            total_demand=('predicted_demand', 'sum'),
            avg_demand=('predicted_demand', 'mean'),
            item_count=('sp_id', 'count')
        ).reset_index()
        
        # Save forecasts to CSVs
        forecast_df.to_csv(os.path.join(output_dir, "item_demand_forecast.csv"), index=False)
        summary_by_loc.to_csv(os.path.join(output_dir, "location_demand_summary.csv"), index=False)
        
        # Save model metrics if any
        if model_metrics:
            with open(os.path.join(output_dir, "model_metrics.json"), 'w') as f:
                json.dump(model_metrics, f, indent=2)
        
        print(f"✅ Forecasts saved to {output_dir}/")
        
        # Create a heatmap of forecasts by location and time
        plt.figure(figsize=(15, 8))
        pivot_data = forecast_df.pivot_table(
            index='loc_id', 
            columns='time_bucket', 
            values='predicted_demand',
            aggfunc='sum'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt='.0f')
        plt.title('Total Demand by Location and Time Period')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "demand_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Upload forecast to feature store
        print("⬆️ Uploading forecast to feature store")
        try:
            forecast_fg = fs.get_or_create_feature_group(
                name="demand_forecast",
                version=1,
                description="Demand forecasts",
                primary_key=['sp_id', 'loc_id', 'time_bucket'],
                event_time='forecast_date',
            )
            forecast_fg.insert(forecast_df)
            print("✅ Forecast uploaded to feature store")
        except Exception as e:
            print(f"⚠️ Could not upload forecast to feature store: {e}")
    else:
        print("⚠️ No forecasts generated")
    
    print("✅ Inference pipeline completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference Pipeline Parameters')
    parser.add_argument('--project', type=str, help='Hopsworks project name')
    parser.add_argument('--model-name', type=str, default='demand_forecaster',
                        help='Base name for the models')
    parser.add_argument('--item', type=int, help='Specific item ID to forecast')
    parser.add_argument('--location', type=int, help='Specific location ID to forecast')
    parser.add_argument('--start-year', type=int, default=2021,
                        help='Start year for forecast')
    parser.add_argument('--start-month', type=int, default=1,
                        help='Start month for forecast')
    parser.add_argument('--periods', type=int, default=12,
                        help='Number of periods to forecast')
    
    args = parser.parse_args()
    
    main(
        project_name=args.project,
        model_name=args.model_name,
        item_id=args.item,
        location_id=args.location,
        start_year=args.start_year,
        start_month=args.start_month,
        periods=args.periods
    )