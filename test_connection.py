import os
import hopsworks
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get connection parameters from environment variables
host = os.getenv("HOST")
port = int(os.getenv("PORT", "443"))
project = os.getenv("PROJECT")
api_key = os.getenv("HOPSWORKS_API_KEY")

try:
    # Connect to Hopsworks
    connection = hopsworks.login(
        host=host,
        port=port, 
        project=project,
        api_key_value=api_key
    )
    print(f"Successfully connected to Hopsworks project: {project}")
    
    # Try to access the feature store as a test
    fs = connection.get_feature_store()
    print(f"Successfully accessed feature store")
    
except Exception as e:
    print(f"Failed to connect to Hopsworks: {str(e)}")