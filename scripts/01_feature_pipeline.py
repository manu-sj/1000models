import pandas as pd
import hopsworks
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_features(df, project_name='models1000', feature_group_name='demand_features', version=1):
    """
    Upload features to Hopsworks feature store
    """
    # Connect to Hopsworks
    project = hopsworks.login(
        host=os.getenv("HOST"),
        port=os.getenv("PORT"),
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=project_name or os.getenv("PROJECT")
    )
    
    # Get feature store
    fs = project.get_feature_store()
    
    # Add datetime column for feature store
    df['datetime'] = datetime.now()
    
    # Create or get feature group
    feature_group = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        description="Item demand by location and time",
        primary_key=['sp_id', 'loc_id', 'time_bucket'],
        event_time='datetime',
    )
    
    # Upload data
    feature_group.insert(df, write_options={"wait_for_job": True})
    
    return feature_group

def load_demand_data(file_path='data/demand_qty_item_loc.csv'):
    """
    Load and preprocess demand data
    """
    demand_df = pd.read_csv(file_path)
    
    # Rename columns for consistency
    demand_df.columns = ['sp_id', 'loc_id', 'time_bucket', 'repetitive_demand_quantity']
    
    return demand_df

def main(project_name='models1000', feature_group_name='demand_features', version=1):
    """
    Main feature pipeline function
    """
    print("🔮 Loading demand data")
    demand_df = load_demand_data()
    
    print(f"🚀 Found {demand_df['sp_id'].nunique()} items × {demand_df['loc_id'].nunique()} locations")
    
    print("⬆️ Uploading to Feature Store")
    upload_features(
        df=demand_df,
        project_name=project_name,
        feature_group_name=feature_group_name,
        version=version
    )
    
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