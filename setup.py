#!/usr/bin/env python3
"""
Concrete Quality Prediction System - Setup Script
Automated setup and initialization for the project
"""

import os
import sys
import subprocess
import urllib.request
import shutil
from pathlib import Path

class ProjectSetup:
    """Setup and initialize the project"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_dirs = [
            'data/raw',
            'data/processed',
            'models',
            'logs',
            'notebooks',
            'src/data_processing',
            'src/models',
            'src/api',
            'src/utils',
            'tests',
            'dashboard',
            'deployment',
            'monitoring/grafana/dashboards',
            'nginx'
        ]
        
    def print_banner(self):
        """Print setup banner"""
        print("=" * 70)
        print("🏗️  CONCRETE QUALITY PREDICTION SYSTEM - SETUP")
        print("=" * 70)
        print()
        
    def check_python_version(self):
        """Check Python version"""
        print("📋 Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 10):
            print("❌ Python 3.10+ is required")
            print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
            sys.exit(1)
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
        print()
        
    def create_directory_structure(self):
        """Create project directories"""
        print("📁 Creating directory structure...")
        for directory in self.required_dirs:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✓ {directory}")
        print()
        
    def create_init_files(self):
        """Create __init__.py files"""
        print("📝 Creating __init__.py files...")
        python_dirs = [
            'src',
            'src/data_processing',
            'src/models',
            'src/api',
            'src/utils',
            'tests'
        ]
        for directory in python_dirs:
            init_file = self.project_root / directory / '__init__.py'
            init_file.touch()
            print(f"   ✓ {directory}/__init__.py")
        print()
        
    def download_dataset(self):
        """Download UCI Concrete dataset"""
        print("📥 Downloading dataset...")
        data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
        output_path = self.project_root / 'data' / 'raw' / 'Concrete_Data.xls'
        
        if output_path.exists():
            print("   ℹ️  Dataset already exists, skipping download")
        else:
            try:
                print("   Downloading from UCI repository...")
                urllib.request.urlretrieve(data_url, output_path)
                print(f"   ✅ Dataset downloaded to {output_path}")
            except Exception as e:
                print(f"   ⚠️  Could not download dataset: {e}")
                print("   Please download manually from:")
                print(f"   {data_url}")
        print()
        
    def create_env_file(self):
        """Create .env template"""
        print("🔧 Creating .env template...")
        env_template = """# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ENVIRONMENT=development
LOG_LEVEL=INFO

# Model Configuration
MODEL_PATH=./models
BEST_MODEL=xgboost

# Database Configuration (optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=concrete_db
DB_USER=concrete_user
DB_PASSWORD=change_this_password

# Redis Configuration (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# Azure ML Configuration (optional)
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_RESOURCE_GROUP=concrete-ml-rg
AZURE_WORKSPACE_NAME=concrete-ml-workspace
AZURE_TENANT_ID=your-tenant-id

# Security
SECRET_KEY=change_this_secret_key_in_production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
GRAFANA_PASSWORD=admin
"""
        env_path = self.project_root / '.env'
        if not env_path.exists():
            with open(env_path, 'w') as f:
                f.write(env_template)
            print("   ✅ .env file created")
            print("   ⚠️  Remember to update credentials before production use!")
        else:
            print("   ℹ️  .env file already exists")
        print()
        
    def create_gitignore(self):
        """Create .gitignore file"""
        print("📝 Creating .gitignore...")
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Models
models/*.pkl
models/*.h5
models/*.pt

# Data
data/raw/*.xls
data/raw/*.xlsx
data/raw/*.csv
data/processed/

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Docker
docker-compose.override.yml

# Azure
.azure/

# Temporary files
*.tmp
*.bak
*.cache
"""
        gitignore_path = self.project_root / '.gitignore'
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("   ✅ .gitignore created")
        print()
        
    def create_docker_files(self):
        """Create Docker configuration files"""
        print("🐳 Creating Docker files...")
        
        # .dockerignore
        dockerignore = """__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
README.md
.env
*.log
data/raw/
data/processed/
notebooks/
tests/
.pytest_cache/
.coverage
htmlcov/
"""
        with open(self.project_root / '.dockerignore', 'w') as f:
            f.write(dockerignore)
        print("   ✅ .dockerignore created")
        print()
        
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")
        print("   This may take several minutes...")
        
        requirements_path = self.project_root / 'requirements.txt'
        if not requirements_path.exists():
            print("   ⚠️  requirements.txt not found")
            print("   Please create requirements.txt first")
            return
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ])
            print("   ✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error installing dependencies: {e}")
            print("   Try running: pip install -r requirements.txt")
        print()
        
    def run_tests(self):
        """Run initial tests"""
        print("🧪 Running initial tests...")
        try:
            # Try importing key libraries
            import numpy
            import pandas
            import sklearn
            import fastapi
            print("   ✅ Core libraries imported successfully")
        except ImportError as e:
            print(f"   ⚠️  Missing library: {e}")
        print()
        
    def create_sample_config(self):
        """Create sample configuration files"""
        print("⚙️  Creating configuration files...")
        
        # Create nginx.conf
        nginx_conf = """events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }
    
    upstream dashboard {
        server dashboard:80;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location / {
            proxy_pass http://dashboard/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
"""
        nginx_path = self.project_root / 'nginx' / 'nginx.conf'
        with open(nginx_path, 'w') as f:
            f.write(nginx_conf)
        print("   ✅ nginx.conf created")
        
        # Create prometheus.yml
        prometheus_yml = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'concrete-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
"""
        prometheus_path = self.project_root / 'monitoring' / 'prometheus.yml'
        with open(prometheus_path, 'w') as f:
            f.write(prometheus_yml)
        print("   ✅ prometheus.yml created")
        print()
        
    def print_next_steps(self):
        """Print next steps for user"""
        print("=" * 70)
        print("✅ SETUP COMPLETE!")
        print("=" * 70)
        print()
        print("📝 Next Steps:")
        print()
        print("1. Activate virtual environment (if not already):")
        print("   python -m venv venv")
        print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print()
        print("2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print()
        print("3. Train the model:")
        print("   python src/models/train.py")
        print("   # Or use the notebook: notebooks/01_training.ipynb")
        print()
        print("4. Start the API server:")
        print("   uvicorn src.api.main:app --reload")
        print()
        print("5. Access the services:")
        print("   - API: http://localhost:8000")
        print("   - API Docs: http://localhost:8000/docs")
        print("   - Dashboard: Open dashboard/index.html in browser")
        print()
        print("6. Docker deployment (optional):")
        print("   docker-compose up -d")
        print()
        print("📚 Additional Resources:")
        print("   - Documentation: README.md")
        print("   - API Examples: See README.md")
        print("   - Azure Deployment: deployment/azure_ml.py")
        print()
        print("🇲🇦 Built for Moroccan Construction Industry")
        print("=" * 70)
        print()
        
    def run_setup(self, skip_install=False):
        """Run complete setup"""
        self.print_banner()
        self.check_python_version()
        self.create_directory_structure()
        self.create_init_files()
        self.download_dataset()
        self.create_env_file()
        self.create_gitignore()
        self.create_docker_files()
        self.create_sample_config()
        
        if not skip_install:
            install = input("Install Python dependencies now? (y/n): ")
            if install.lower() == 'y':
                self.install_dependencies()
        
        self.run_tests()
        self.print_next_steps()


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Setup Concrete Quality Prediction System"
    )
    parser.add_argument(
        '--skip-install',
        action='store_true',
        help='Skip dependency installation'
    )
    parser.add_argument(
        '--minimal',
        action='store_true',
        help='Minimal setup (directories only)'
    )
    
    args = parser.parse_args()
    
    setup = ProjectSetup()
    
    if args.minimal:
        print("Running minimal setup...")
        setup.print_banner()
        setup.create_directory_structure()
        setup.create_init_files()
        print("✅ Minimal setup complete!")
    else:
        setup.run_setup(skip_install=args.skip_install)


if __name__ == "__main__":
    main()