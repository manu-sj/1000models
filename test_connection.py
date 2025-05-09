import os
import hopsworks
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Connect to Hopsworks
project = hopsworks.login(
    host=os.getenv("HOST"),
    port=os.getenv("PORT"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("PROJECT")
)

print(f"Successfully connected to Hopsworks project: {os.getenv('PROJECT')}")

# Try to access the feature store as a test
fs = project.get_feature_store()
print(f"Successfully accessed feature store")