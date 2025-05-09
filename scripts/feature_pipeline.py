import pandas as pd
import hopsworks
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main(project_name='models1000', feature_group_name='demand_features', version=1):
    """
    Feature pipeline to process demand data and upload to feature store
    """
    print("🔮 Connecting to Hopsworks Feature Store")
    api_key = os.getenv("HOPSWORKS_API_KEY")
    host = os.getenv("HOST")
    port = os.getenv("PORT")

    project_name = project_name or os.getenv("PROJECT")
    project = hopsworks.login(host=host, port=port, api_key_value=api_key, project=project_name)
    
    fs = project.get_feature_store()
    
    print("📊 Loading source data")
    demand_df = pd.read_csv('data/demand_qty_item_loc.csv')
    
    # Convert column headers to match the data
    demand_df.columns = ['sp_id', 'loc_id', 'time_bucket', 'repetitive_demand_quantity']
    
    # Add datetime column for feature store
    demand_df['datetime'] = datetime.now()
    
    print(f"🚀 Found {demand_df['sp_id'].nunique()} unique items")
    print(f"🚀 Found {demand_df['loc_id'].nunique()} unique locations")
    print(f"🚀 Found {demand_df['time_bucket'].nunique()} unique time periods")
    
    print("⬆️ Creating/getting feature group")
    # Define the feature group
    demand_fg = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        description="Item demand by location and time",
        primary_key=['sp_id', 'loc_id', 'time_bucket'],
        event_time='datetime',
    )
    
    print("⬆️ Uploading data to the Feature Store")
    demand_fg.insert(demand_df, write_options={"wait_for_job": True})
    
    print("✅ Feature pipeline completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Pipeline Parameters')
    parser.add_argument('--project', type=str, help='Hopsworks project name')
    parser.add_argument('--feature-group', type=str, default='demand_features', 
                        help='Feature group name')
    parser.add_argument('--version', type=int, default=1, 
                        help='Feature group version')
    
    args = parser.parse_args()
    
    main(
        project_name=args.project,
        feature_group_name=args.feature_group,
        version=args.version
    )