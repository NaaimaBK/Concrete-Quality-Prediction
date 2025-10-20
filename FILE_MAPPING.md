# File Mapping Guide

Copy the artifacts from Claude's responses into these files:

## Core Application Files

1. **src/data_processing/etl.py**
   - Copy content from artifact: "Data Processing Pipeline (ETL)"
   
2. **src/models/train.py**
   - Copy content from artifact: "ML Models Training & Evaluation"
   
3. **src/api/main.py**
   - Copy content from artifact: "FastAPI Deployment Service"
   
4. **dashboard/ConcreteApp.jsx**
   - Copy content from artifact: "Interactive Concrete Strength Dashboard"

## Configuration Files

5. **Dockerfile**
   - Copy content from artifact: "Docker Deployment Configuration"
   
6. **docker-compose.yml**
   - Copy content from artifact: "Docker Compose Configuration"
   
7. **requirements.txt**
   - Copy content from artifact: "Python Requirements"

## Deployment Files

8. **deployment/azure_ml.py**
   - Copy content from artifact: "Azure ML Training & Deployment"
   
9. **setup.py**
   - Copy content from artifact: "Project Setup Script"

## Documentation

10. **README.md**
    - Copy content from artifact: "Project README"
    
11. **docs/AZURE_DEPLOYMENT.md**
    - Copy content from artifact: "Azure Deployment Guide"

## Notebooks

12. **notebooks/training.ipynb**
    - Copy content from artifact: "Complete Training Notebook"
    - Convert the Python code to Jupyter notebook format

## Quick Copy Instructions

For each artifact:
1. Find the artifact in Claude's conversation
2. Copy the entire code block
3. Paste into the corresponding file listed above
4. Save the file

After copying all files, run:
```bash
python setup.py
pip install -r requirements.txt
```

Then follow the README.md for usage instructions.
