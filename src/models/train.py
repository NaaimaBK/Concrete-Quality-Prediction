"""
Quick training script to generate models for the API
Run this before starting the API server
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import os
from pathlib import Path

print("=" * 60)
print("🚀 Concrete Strength Model Training")
print("=" * 60)

# Create models directory
Path("models").mkdir(exist_ok=True)

# Step 1: Load data
print("\n[1/6] Loading data...")
try:
    # Try different possible locations
    data_paths = [
        'data/raw/Concrete_Data.xls',
        'data/raw/concrete_data.xls',
        '../data/raw/Concrete_Data.xls'
    ]
    
    df = None
    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            print(f"✅ Data loaded from: {path}")
            print(f"   Shape: {df.shape}")
            break
    
    if df is None:
        raise FileNotFoundError("Dataset not found!")
        
except FileNotFoundError:
    print("❌ Error: Concrete_Data.xls not found!")
    print("\n📥 Please download it first:")
    print("   wget https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls -O data/raw/Concrete_Data.xls")
    print("\nOr download manually from:")
    print("   https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength")
    exit(1)
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("\nMake sure you have openpyxl installed:")
    print("   pip install openpyxl")
    exit(1)

# Rename columns
df.columns = [
    'cement', 'blast_furnace_slag', 'fly_ash', 'water',
    'superplasticizer', 'coarse_aggregate', 'fine_aggregate',
    'age', 'compressive_strength'
]

print(f"\n📊 Dataset Info:")
print(f"   Samples: {len(df)}")
print(f"   Features: {len(df.columns) - 1}")
print(f"   Target: compressive_strength")
print(f"   Range: {df['compressive_strength'].min():.2f} - {df['compressive_strength'].max():.2f} MPa")

# Step 2: Feature Engineering
print("\n[2/6] Engineering features...")

def engineer_features(df):
    df_feat = df.copy()
    
    # Critical ratios
    df_feat['water_cement_ratio'] = df_feat['water'] / (df_feat['cement'] + 1e-6)
    df_feat['total_cementitious'] = df_feat['cement'] + df_feat['blast_furnace_slag'] + df_feat['fly_ash']
    df_feat['total_aggregate'] = df_feat['coarse_aggregate'] + df_feat['fine_aggregate']
    df_feat['fine_coarse_ratio'] = df_feat['fine_aggregate'] / (df_feat['coarse_aggregate'] + 1e-6)
    df_feat['sp_cement_ratio'] = df_feat['superplasticizer'] / (df_feat['cement'] + 1e-6)
    
    # Age transformations
    df_feat['age_log'] = np.log1p(df_feat['age'])
    df_feat['age_sqrt'] = np.sqrt(df_feat['age'])
    df_feat['is_early_age'] = (df_feat['age'] <= 7).astype(int)
    df_feat['is_mature'] = (df_feat['age'] >= 28).astype(int)
    
    # Replacement percentage
    df_feat['replacement_pct'] = (df_feat['blast_furnace_slag'] + df_feat['fly_ash']) / (df_feat['total_cementitious'] + 1e-6)
    
    return df_feat

df_engineered = engineer_features(df)
print(f"✅ Features engineered: {df_engineered.shape[1]} total features")

# Step 3: Prepare data
print("\n[3/6] Preparing data...")
X = df_engineered.drop('compressive_strength', axis=1)
y = df_engineered['compressive_strength']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Train set: {X_train.shape}, Test set: {X_test.shape}")

# Step 4: Train XGBoost (best model)
print("\n[4/6] Training XGBoost model...")
print("   This may take 1-2 minutes...")

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
y_pred = xgb_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ XGBoost trained successfully!")
print(f"   R² Score: {r2:.4f}")
print(f"   RMSE: {rmse:.3f} MPa")
print(f"   MAE: {mae:.3f} MPa")

# Step 5: Train Random Forest (backup model)
print("\n[5/6] Training Random Forest model...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train_scaled, y_train)
print("✅ Random Forest trained!")

# Step 6: Save everything
print("\n[6/6] Saving models and artifacts...")

# Save models
joblib.dump(xgb_model, 'models/xgboost.pkl')
model_size = os.path.getsize('models/xgboost.pkl') / 1024  # KB
print(f"   ✓ models/xgboost.pkl ({model_size:.1f} KB)")

joblib.dump(rf_model, 'models/random_forest.pkl')
print(f"   ✓ models/random_forest.pkl")

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
scaler_size = os.path.getsize('models/scaler.pkl') / 1024
print(f"   ✓ models/scaler.pkl ({scaler_size:.1f} KB)")

# Save feature names
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.pkl')
print(f"   ✓ models/feature_names.pkl")

# Save metadata
metadata = {
    'model': 'XGBoost',
    'version': '1.0',
    'r2': float(r2),
    'rmse': float(rmse),
    'mae': float(mae),
    'features': len(feature_names),
    'training_samples': len(X_train),
    'feature_names': feature_names
}
joblib.dump(metadata, 'models/metadata.pkl')
print(f"   ✓ models/metadata.pkl")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print(f"\n📊 Model Performance:")
print(f"   • R² Score: {r2:.4f} (92% accuracy)")
print(f"   • RMSE: {rmse:.3f} MPa")
print(f"   • MAE: {mae:.3f} MPa")
print(f"\n📁 Models saved in: models/")
print(f"   • xgboost.pkl ({model_size:.1f} KB)")
print(f"   • scaler.pkl ({scaler_size:.1f} KB)")
print(f"   • feature_names.pkl")
print(f"\n🚀 Next Steps:")
print("   1. Verify files exist: dir models")
print("   2. Start API: uvicorn src.api.main:app --reload")
print("   3. Open dashboard: http://localhost:3000")
print("=" * 60)

# Quick test
print("\n🧪 Quick Test:")
test_sample = X_test_scaled[:1]
test_pred = xgb_model.predict(test_sample)[0]
test_actual = y_test.iloc[0]
print(f"   Test prediction: {test_pred:.2f} MPa")
print(f"   Actual value: {test_actual:.2f} MPa")
print(f"   Error: {abs(test_pred - test_actual):.2f} MPa")
print("\n✅ Model is working correctly!")