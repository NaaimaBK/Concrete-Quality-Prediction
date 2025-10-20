# Azure ML Deployment Guide

Complete guide for deploying the Concrete Quality Prediction System on Azure.

## 📋 Prerequisites

- Azure subscription
- Azure CLI installed
- Python 3.10+
- Docker (for container deployment)
- Trained models ready

## 🚀 Deployment Options

### Option 1: Azure ML Managed Endpoint (Recommended)
### Option 2: Azure Container Instances
### Option 3: Azure Kubernetes Service (AKS)
### Option 4: Azure App Service

---

## Option 1: Azure ML Managed Endpoint

### Step 1: Install Azure ML SDK

```bash
pip install azure-ai-ml azure-identity
```

### Step 2: Configure Azure CLI

```bash
# Login to Azure
az login

# Set default subscription
az account set --subscription "your-subscription-id"

# Create resource group
az group create --name concrete-ml-rg --location westeurope
```

### Step 3: Create Azure ML Workspace

```bash
# Create workspace
az ml workspace create \
    --name concrete-ml-workspace \
    --resource-group concrete-ml-rg \
    --location westeurope
```

### Step 4: Deploy Model

```python
# Use the provided script
python deployment/azure_ml.py

# Or manually:
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id="your-subscription-id",
    resource_group_name="concrete-ml-rg",
    workspace_name="concrete-ml-workspace"
)
```

### Step 5: Test Endpoint

```python
import requests
import json

endpoint_url = "https://your-endpoint.westeurope.inference.ml.azure.com/score"
api_key = "your-api-key"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "data": [{
        "cement": 300,
        "blast_furnace_slag": 50,
        "fly_ash": 0,
        "water": 180,
        "superplasticizer": 5,
        "coarse_aggregate": 1000,
        "fine_aggregate": 750,
        "age": 28
    }]
}

response = requests.post(endpoint_url, json=data, headers=headers)
print(response.json())
```

---

## Option 2: Azure Container Instances (ACI)

### Step 1: Build Docker Image

```bash
# Build image
docker build -t concrete-api:latest .

# Test locally
docker run -p 8000:8000 concrete-api:latest
```

### Step 2: Push to Azure Container Registry

```bash
# Create ACR
az acr create \
    --resource-group concrete-ml-rg \
    --name concreteacr \
    --sku Basic

# Login to ACR
az acr login --name concreteacr

# Tag image
docker tag concrete-api:latest concreteacr.azurecr.io/concrete-api:v1

# Push image
docker push concreteacr.azurecr.io/concrete-api:v1
```

### Step 3: Deploy to ACI

```bash
# Enable admin access
az acr update --name concreteacr --admin-enabled true

# Get credentials
az acr credential show --name concreteacr

# Deploy container
az container create \
    --resource-group concrete-ml-rg \
    --name concrete-api \
    --image concreteacr.azurecr.io/concrete-api:v1 \
    --cpu 2 \
    --memory 4 \
    --registry-login-server concreteacr.azurecr.io \
    --registry-username <username> \
    --registry-password <password> \
    --dns-name-label concrete-api-unique \
    --ports 8000
```

### Step 4: Access API

```bash
# Get FQDN
az container show \
    --resource-group concrete-ml-rg \
    --name concrete-api \
    --query ipAddress.fqdn

# API available at: http://<fqdn>:8000
```

---

## Option 3: Azure Kubernetes Service (AKS)

### Step 1: Create AKS Cluster

```bash
# Create AKS cluster
az aks create \
    --resource-group concrete-ml-rg \
    --name concrete-aks-cluster \
    --node-count 2 \
    --node-vm-size Standard_DS2_v2 \
    --enable-addons monitoring \
    --generate-ssh-keys

# Get credentials
az aks get-credentials \
    --resource-group concrete-ml-rg \
    --name concrete-aks-cluster
```

### Step 2: Create Kubernetes Manifests

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: concrete-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: concrete-api
  template:
    metadata:
      labels:
        app: concrete-api
    spec:
      containers:
      - name: concrete-api
        image: concreteacr.azurecr.io/concrete-api:v1
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: ENVIRONMENT
          value: "production"
---
apiVersion: v1
kind: Service
metadata:
  name: concrete-api-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: concrete-api
```

### Step 3: Deploy to AKS

```bash
# Apply deployment
kubectl apply -f deployment.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl get services

# Get external IP
kubectl get service concrete-api-service
```

---

## Option 4: Azure App Service

### Step 1: Create App Service Plan

```bash
# Create App Service Plan
az appservice plan create \
    --name concrete-app-plan \
    --resource-group concrete-ml-rg \
    --sku B1 \
    --is-linux

# Create Web App
az webapp create \
    --resource-group concrete-ml-rg \
    --plan concrete-app-plan \
    --name concrete-api-app \
    --deployment-container-image-name concreteacr.azurecr.io/concrete-api:v1
```

### Step 2: Configure Container Settings

```bash
# Set container registry credentials
az webapp config container set \
    --name concrete-api-app \
    --resource-group concrete-ml-rg \
    --docker-custom-image-name concreteacr.azurecr.io/concrete-api:v1 \
    --docker-registry-server-url https://concreteacr.azurecr.io \
    --docker-registry-server-user <username> \
    --docker-registry-server-password <password>

# Set environment variables
az webapp config appsettings set \
    --resource-group concrete-ml-rg \
    --name concrete-api-app \
    --settings ENVIRONMENT=production MODEL_PATH=/app/models
```

### Step 3: Enable Continuous Deployment

```bash
# Enable CI/CD
az webapp deployment container config \
    --name concrete-api-app \
    --resource-group concrete-ml-rg \
    --enable-cd true
```

---

## 🔐 Security Best Practices

### 1. Use Managed Identity

```bash
# Enable system-assigned managed identity
az webapp identity assign \
    --name concrete-api-app \
    --resource-group concrete-ml-rg
```

### 2. Configure Key Vault

```bash
# Create Key Vault
az keyvault create \
    --name concrete-keyvault \
    --resource-group concrete-ml-rg \
    --location westeurope

# Store secrets
az keyvault secret set \
    --vault-name concrete-keyvault \
    --name db-password \
    --value "your-secure-password"
```

### 3. Enable SSL/TLS

```bash
# Add custom domain and SSL
az webapp config hostname add \
    --webapp-name concrete-api-app \
    --resource-group concrete-ml-rg \
    --hostname www.yourcompany.com

# Enable HTTPS only
az webapp update \
    --resource-group concrete-ml-rg \
    --name concrete-api-app \
    --https-only true
```

---

## 📊 Monitoring & Logging

### Application Insights

```bash
# Create Application Insights
az monitor app-insights component create \
    --app concrete-api-insights \
    --location westeurope \
    --resource-group concrete-ml-rg

# Get instrumentation key
az monitor app-insights component show \
    --app concrete-api-insights \
    --resource-group concrete-ml-rg \
    --query instrumentationKey
```

### Configure in Application

```python
# Add to your FastAPI app
from applicationinsights.flask import AppInsights

app.config['APPINSIGHTS_INSTRUMENTATIONKEY'] = 'your-key'
app_insights = AppInsights(app)
```

---

## 💰 Cost Optimization

### 1. Use Spot Instances (AKS)

```bash
az aks nodepool add \
    --resource-group concrete-ml-rg \
    --cluster-name concrete-aks-cluster \
    --name spotnodepool \
    --priority Spot \
    --eviction-policy Delete \
    --spot-max-price -1 \
    --node-count 2
```

### 2. Auto-scaling

```yaml
# HorizontalPodAutoscaler for AKS
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: concrete-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: concrete-api
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### 3. Reserved Instances

Consider purchasing Azure Reserved VM Instances for long-term deployments (1-3 years) to save up to 72%.

---

## 🔄 CI/CD Pipeline

### Azure DevOps Pipeline (azure-pipelines.yml)

```yaml
trigger:
  branches:
    include:
    - main

pool:
  vmImage: 'ubuntu-latest'

variables:
  dockerRegistryServiceConnection: 'concreteacr'
  imageRepository: 'concrete-api'
  containerRegistry: 'concreteacr.azurecr.io'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and Push
  jobs:
  - job: Build
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: Dockerfile
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Deploy
  displayName: Deploy to AKS
  jobs:
  - deployment: Deploy
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: KubernetesManifest@0
            displayName: Deploy to Kubernetes
            inputs:
              action: deploy
              manifests: |
                deployment.yaml
              containers: |
                $(containerRegistry)/$(imageRepository):$(tag)
```

---

## 🧪 Testing Deployment

### Health Check

```bash
# Test health endpoint
curl http://your-endpoint/health

# Expected response:
# {"status": "healthy", "model_loaded": true}
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test
ab -n 1000 -c 10 -T application/json -p test_data.json \
   http://your-endpoint/predict
```

### Smoke Tests

```python
import requests

def test_deployment():
    url = "http://your-endpoint/predict"
    data = {
        "cement": 300,
        "blast_furnace_slag": 50,
        "fly_ash": 0,
        "water": 180,
        "superplasticizer": 5,
        "coarse_aggregate": 1000,
        "fine_aggregate": 750,
        "age": 28
    }
    
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert 'predicted_strength' in response.json()
    print("✅ Deployment test passed!")

test_deployment()
```

---

## 📈 Scaling Strategies

### Horizontal Scaling
- AKS: Increase pod replicas
- ACI: Deploy multiple container instances
- App Service: Scale out instances

### Vertical Scaling
- Increase CPU/Memory resources
- Upgrade App Service plan tier

### Auto-scaling Rules

```bash
# Azure App Service auto-scale
az monitor autoscale create \
    --resource-group concrete-ml-rg \
    --resource concrete-api-app \
    --resource-type Microsoft.Web/serverfarms \
    --name autoscale-rules \
    --min-count 1 \
    --max-count 10 \
    --count 2

# Add CPU-based rule
az monitor autoscale rule create \
    --resource-group concrete-ml-rg \
    --autoscale-name autoscale-rules \
    --scale out 1 \
    --condition "Percentage CPU > 70 avg 5m"
```

---

## 🆘 Troubleshooting

### Common Issues

**1. Container won't start:**
```bash
# Check logs
az container logs \
    --resource-group concrete-ml-rg \
    --name concrete-api

# Check events
kubectl describe pod <pod-name>
```

**2. Model not loading:**
- Verify model files in container
- Check file permissions
- Ensure sufficient memory

**3. High latency:**
- Enable caching
- Optimize model inference
- Use batch predictions

---

## 📞 Support

For Azure-specific issues:
- Azure Support: https://azure.microsoft.com/support/
- Azure ML Documentation: https://docs.microsoft.com/azure/machine-learning/

For project issues:
- GitHub Issues: [link]
- Email: support@yourcompany.com

---

**🇲🇦 Deployed for Moroccan Construction Industry**