"""
Azure ML Training and Deployment Script
Trains models on Azure ML and deploys as REST endpoint
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AzureMLDeployment:
    """Handle Azure ML model training and deployment"""
    
    def __init__(self, subscription_id, resource_group, workspace_name):
        """Initialize Azure ML client"""
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        logger.info(f"Connected to workspace: {workspace_name}")
    
    def create_environment(self):
        """Create Azure ML environment with dependencies"""
        env = Environment(
            name="concrete-prediction-env",
            description="Environment for concrete strength prediction",
            conda_file="conda.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        
        env = self.ml_client.environments.create_or_update(env)
        logger.info(f"Environment created: {env.name}")
        return env
    
    def register_model(self, model_path, model_name="concrete-strength-model"):
        """Register trained model in Azure ML"""
        model = Model(
            path=model_path,
            type=AssetTypes.CUSTOM_MODEL,
            name=model_name,
            description="XGBoost model for concrete compressive strength prediction",
            tags={
                "framework": "XGBoost",
                "task": "regression",
                "industry": "construction",
                "country": "Morocco"
            }
        )
        
        registered_model = self.ml_client.models.create_or_update(model)
        logger.info(f"Model registered: {registered_model.name} v{registered_model.version}")
        return registered_model
    
    def create_endpoint(self, endpoint_name="concrete-strength-endpoint"):
        """Create online endpoint for model deployment"""
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Endpoint for concrete strength prediction",
            auth_mode="key",
            tags={
                "project": "concrete-quality",
                "environment": "production"
            }
        )
        
        endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint created: {endpoint.name}")
        return endpoint
    
    def deploy_model(self, endpoint_name, model, environment):
        """Deploy model to endpoint"""
        deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=endpoint_name,
            model=model,
            environment=environment,
            code_configuration=CodeConfiguration(
                code="./src",
                scoring_script="score.py"
            ),
            instance_type="Standard_DS3_v2",
            instance_count=1,
            liveness_probe={
                "initial_delay": 10,
                "period": 10,
                "timeout": 2,
                "success_threshold": 1,
                "failure_threshold": 3
            },
            readiness_probe={
                "initial_delay": 10,
                "period": 10,
                "timeout": 2,
                "success_threshold": 1,
                "failure_threshold": 3
            }
        )
        
        deployment = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        logger.info(f"Model deployed: {deployment.name}")
        
        # Set traffic to 100% for this deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {"blue": 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        return deployment
    
    def get_endpoint_keys(self, endpoint_name):
        """Get endpoint authentication keys"""
        keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
        logger.info("Retrieved endpoint keys")
        return keys
    
    def test_endpoint(self, endpoint_name, sample_data):
        """Test deployed endpoint"""
        result = self.ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=sample_data
        )
        logger.info(f"Endpoint test result: {result}")
        return result
    
    def full_deployment_pipeline(self, model_path, endpoint_name="concrete-endpoint"):
        """Execute complete deployment pipeline"""
        logger.info("=" * 60)
        logger.info("Starting Azure ML Deployment Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Create environment
        logger.info("\n[1/5] Creating environment...")
        env = self.create_environment()
        
        # Step 2: Register model
        logger.info("\n[2/5] Registering model...")
        model = self.register_model(model_path)
        
        # Step 3: Create endpoint
        logger.info("\n[3/5] Creating endpoint...")
        endpoint = self.create_endpoint(endpoint_name)
        
        # Step 4: Deploy model
        logger.info("\n[4/5] Deploying model...")
        deployment = self.deploy_model(endpoint_name, model, env)
        
        # Step 5: Get keys
        logger.info("\n[5/5] Retrieving endpoint keys...")
        keys = self.get_endpoint_keys(endpoint_name)
        
        logger.info("\n" + "=" * 60)
        logger.info("Deployment Complete!")
        logger.info("=" * 60)
        logger.info(f"Endpoint Name: {endpoint_name}")
        logger.info(f"Scoring URI: {endpoint.scoring_uri}")
        logger.info(f"Primary Key: {keys.primary_key[:20]}...")
        logger.info("=" * 60)
        
        return {
            "endpoint": endpoint,
            "deployment": deployment,
            "keys": keys
        }

# Scoring script for Azure ML (save as score.py)
SCORING_SCRIPT = '''
import json
import joblib
import numpy as np
import pandas as pd
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

def init():
    """Initialize model and preprocessing objects"""
    global model, scaler, feature_names
    
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    feature_names = [
        'cement', 'blast_furnace_slag', 'fly_ash', 'water',
        'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age',
        'water_cement_ratio', 'total_cementitious', 'total_aggregate',
        'fine_coarse_ratio', 'sp_cement_ratio', 'age_log', 'age_sqrt',
        'is_early_age', 'is_mature', 'replacement_pct'
    ]

def engineer_features(df):
    """Engineer features from raw input"""
    df['water_cement_ratio'] = df['water'] / (df['cement'] + 1e-6)
    df['total_cementitious'] = df['cement'] + df['blast_furnace_slag'] + df['fly_ash']
    df['total_aggregate'] = df['coarse_aggregate'] + df['fine_aggregate']
    df['fine_coarse_ratio'] = df['fine_aggregate'] / (df['coarse_aggregate'] + 1e-6)
    df['sp_cement_ratio'] = df['superplasticizer'] / (df['cement'] + 1e-6)
    df['age_log'] = np.log1p(df['age'])
    df['age_sqrt'] = np.sqrt(df['age'])
    df['is_early_age'] = (df['age'] <= 7).astype(int)
    df['is_mature'] = (df['age'] >= 28).astype(int)
    df['replacement_pct'] = (df['blast_furnace_slag'] + df['fly_ash']) / (df['total_cementitious'] + 1e-6)
    return df

@input_schema('data', PandasParameterType(pd.DataFrame({
    "cement": [300],
    "blast_furnace_slag": [50],
    "fly_ash": [0],
    "water": [180],
    "superplasticizer": [5],
    "coarse_aggregate": [1000],
    "fine_aggregate": [750],
    "age": [28]
})))
@output_schema(NumpyParameterType(np.array([35.5])))
def run(data):
    """Score the model"""
    try:
        # Engineer features
        data_feat = engineer_features(data)
        data_feat = data_feat[feature_names]
        
        # Scale features
        data_scaled = scaler.transform(data_feat)
        
        # Predict
        predictions = model.predict(data_scaled)
        
        return predictions.tolist()
    except Exception as e:
        error = str(e)
        return {"error": error}
'''

# Example usage
if __name__ == "__main__":
    # Configuration
    SUBSCRIPTION_ID = "your-subscription-id"
    RESOURCE_GROUP = "concrete-ml-rg"
    WORKSPACE_NAME = "concrete-ml-workspace"
    
    # Initialize deployment
    azure_deploy = AzureMLDeployment(
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    # Deploy model
    # result = azure_deploy.full_deployment_pipeline(
    #     model_path="./models",
    #     endpoint_name="concrete-strength-api"
    # )
    
    print("Azure ML deployment script ready!")
    print("\nSave the SCORING_SCRIPT content as 'score.py' for Azure ML deployment")