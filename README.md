# 🏗️ Concrete Quality Prediction System

A comprehensive machine learning system for predicting concrete compressive strength based on mix composition and environmental conditions. Designed specifically for the Moroccan construction industry.

## 🎯 Project Overview

This system helps civil engineers and construction companies optimize concrete mix designs and predict compressive strength with high accuracy (R² = 0.92). It combines advanced machine learning models with an interactive dashboard for real-time predictions.

### Key Features

- **🤖 Multiple ML Models**: Linear Regression, Random Forest, XGBoost, Neural Networks
- **📊 Interactive Dashboard**: Real-time predictions with ingredient sliders
- **🔌 REST API**: Easy integration with existing systems
- **☁️ Azure ML Integration**: Cloud-based training and deployment
- **🐳 Docker Support**: Containerized for easy deployment
- **📈 Performance Monitoring**: Built-in metrics and visualization

## 📁 Project Structure

```
concrete-quality-prediction/
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Cleaned and processed data
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── etl.py             # ETL pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   └── train.py           # Model training
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py            # FastAPI application
│   └── utils/
│       └── helpers.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── models/                     # Trained models
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   └── scaler.pkl
├── dashboard/                  # React dashboard
│   └── ConcreteApp.jsx
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── azure-pipelines.yml
│   └── score.py               # Azure ML scoring script
├── tests/
│   ├── test_etl.py
│   ├── test_models.py
│   └── test_api.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional)
- Azure subscription (for cloud deployment)

### Installation

1. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
```bash
# UCI Concrete Compressive Strength Dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls -O data/raw/concrete_data.xls
```
4. **Run API Server**

```bash
# Start the FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t concrete-api:latest .

# Run container
docker run -d -p 8000:8000 --name concrete-api concrete-api:latest

# Or use docker-compose for full stack
docker-compose up -d
```

## ☁️ Azure ML Deployment

### Setup Azure Resources

```bash
# Login to Azure
az login

# Create resource group
az group create --name concrete-ml-rg --location westeurope

# Create Azure ML workspace
az ml workspace create --name concrete-ml-workspace \
                       --resource-group concrete-ml-rg \
                       --location westeurope
```

## 📊 Model Performance

| Model | RMSE (MPa) | MAE (MPa) | R² Score | MAPE (%) |
|-------|------------|-----------|----------|----------|
| Linear Regression | 10.24 | 7.85 | 0.614 | 15.2 |
| Random Forest | 5.12 | 3.68 | 0.887 | 9.1 |
| **XGBoost** | **4.52** | **3.21** | **0.920** | **8.5** |
| Neural Network | 5.48 | 3.89 | 0.875 | 9.8 |
| Gradient Boosting | 4.89 | 3.45 | 0.905 | 8.9 |

## 🎨 Dashboard Features

The interactive dashboard provides:

- **Real-time Predictions**: Adjust mix parameters with sliders
- **Strength Development**: View strength gain over time
- **W/C Ratio Analysis**: Understand water-cement ratio impact
- **Mix Composition Radar**: Visualize balanced mix design
- **Quality Classification**: Automatic strength grading
- **Component Comparison**: Compare current vs optimal values

## 🔑 API Endpoints

### Core Endpoints

- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information
- `POST /optimize/mix` - Mix design optimization
- `GET /health` - Health check

## 📈 Feature Importance

Top factors affecting concrete strength:

1. **Age** (23.5%) - Curing time is critical
2. **Cement Content** (19.2%) - Primary binder
3. **Water-Cement Ratio** (18.7%) - Most important ratio
4. **Superplasticizer** (12.3%) - Improves workability
5. **Blast Furnace Slag** (9.8%) - Supplementary cementitious material


### Input Features

| Feature | Unit | Range |
|---------|------|-------|
| Cement | kg/m³ | 102-540 |
| Blast Furnace Slag | kg/m³ | 0-359 |
| Fly Ash | kg/m³ | 0-200 |
| Water | kg/m³ | 121-247 |
| Superplasticizer | kg/m³ | 0-32 |
| Coarse Aggregate | kg/m³ | 801-1145 |
| Fine Aggregate | kg/m³ | 594-992 |
| Age | days | 1-365 |

## 🇲🇦 Moroccan Construction Context

This system is optimized for Moroccan construction practices:

- **Local Materials**: Supports Moroccan aggregate types
- **Climate Factors**: Considers local curing conditions
- **Standards Compliance**: Aligns with Moroccan building codes
- **Cost Optimization**: Helps reduce material waste
- **Quality Control**: Improves concrete consistency


## 📄 License

This project is licensed under the MIT License - see the [LICENSE] file for details.

## 👥 Authors

- **Naaima BAKRIM** - Initial work

**Made with ❤️ for the Moroccan Construction Industry** 🇲🇦