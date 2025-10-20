from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Concrete Compressive Strength Prediction API",
    description="API for predicting concrete compressive strength based on mix composition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ConcreteMix(BaseModel):
    """Concrete mix composition input"""
    cement: float = Field(..., ge=0, le=600, description="Cement (kg/m³)")
    blast_furnace_slag: float = Field(0, ge=0, le=400, description="Blast Furnace Slag (kg/m³)")
    fly_ash: float = Field(0, ge=0, le=300, description="Fly Ash (kg/m³)")
    water: float = Field(..., ge=100, le=300, description="Water (kg/m³)")
    superplasticizer: float = Field(0, ge=0, le=50, description="Superplasticizer (kg/m³)")
    coarse_aggregate: float = Field(..., ge=700, le=1200, description="Coarse Aggregate (kg/m³)")
    fine_aggregate: float = Field(..., ge=500, le=1000, description="Fine Aggregate (kg/m³)")
    age: int = Field(..., ge=1, le=365, description="Curing age (days)")
    
    @validator('cement')
    def validate_cement(cls, v):
        if v < 100:
            raise ValueError("Cement content too low for structural concrete")
        return v
    
    @validator('water', 'cement')
    def validate_water_cement_ratio(cls, v, values):
        if 'cement' in values and 'water' in values:
            wc_ratio = values.get('water', v) / values.get('cement', v)
            if wc_ratio < 0.2 or wc_ratio > 1.0:
                logger.warning(f"Unusual water-cement ratio: {wc_ratio:.2f}")
        return v

class PredictionResponse(BaseModel):
    """Prediction response with confidence intervals"""
    predicted_strength: float = Field(..., description="Predicted compressive strength (MPa)")
    confidence_interval_lower: Optional[float] = Field(None, description="Lower bound (MPa)")
    confidence_interval_upper: Optional[float] = Field(None, description="Upper bound (MPa)")
    model_used: str = Field(..., description="Model name used for prediction")
    input_features: Dict = Field(..., description="Input features used")
    quality_assessment: str = Field(..., description="Concrete quality classification")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    mixes: List[ConcreteMix]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict

class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    version: str
    accuracy_metrics: Dict
    feature_names: List[str]

# Global variables for models
loaded_model = None
scaler = None
feature_names = None
model_metrics = None

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global loaded_model, scaler, feature_names, model_metrics
    
    try:
        # Load best model (change path as needed)
        loaded_model = joblib.load('models/xgboost.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Load feature names
        feature_names = [
            'cement', 'blast_furnace_slag', 'fly_ash', 'water',
            'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age',
            'water_cement_ratio', 'total_cementitious', 'total_aggregate',
            'fine_coarse_ratio', 'sp_cement_ratio', 'age_log', 'age_sqrt',
            'is_early_age', 'is_mature', 'replacement_pct'
        ]
        
        # Load metrics (mock data - replace with actual)
        model_metrics = {
            'rmse': 4.52,
            'mae': 3.21,
            'r2': 0.92,
            'mape': 8.5
        }
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        logger.info("Running without pre-trained models (demo mode)")

def engineer_features(mix_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw mix data"""
    df = mix_data.copy()
    
    # Water-cement ratio
    df['water_cement_ratio'] = df['water'] / (df['cement'] + 1e-6)
    
    # Total cementitious
    df['total_cementitious'] = df['cement'] + df['blast_furnace_slag'] + df['fly_ash']
    
    # Total aggregates
    df['total_aggregate'] = df['coarse_aggregate'] + df['fine_aggregate']
    
    # Fine to coarse ratio
    df['fine_coarse_ratio'] = df['fine_aggregate'] / (df['coarse_aggregate'] + 1e-6)
    
    # Superplasticizer ratio
    df['sp_cement_ratio'] = df['superplasticizer'] / (df['cement'] + 1e-6)
    
    # Age transformations
    df['age_log'] = np.log1p(df['age'])
    df['age_sqrt'] = np.sqrt(df['age'])
    df['is_early_age'] = (df['age'] <= 7).astype(int)
    df['is_mature'] = (df['age'] >= 28).astype(int)
    
    # Replacement percentage
    df['replacement_pct'] = (df['blast_furnace_slag'] + df['fly_ash']) / (df['total_cementitious'] + 1e-6)
    
    return df

def classify_strength(strength: float) -> str:
    """Classify concrete quality based on compressive strength"""
    if strength < 20:
        return "Low Strength - Non-structural use only"
    elif strength < 30:
        return "Moderate Strength - Light structural applications"
    elif strength < 40:
        return "Good Strength - Standard structural concrete"
    elif strength < 50:
        return "High Strength - Heavy structural applications"
    else:
        return "Very High Strength - Special structural applications"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Concrete Compressive Strength Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Single prediction",
            "/predict/batch": "Batch predictions",
            "/model/info": "Model information",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": loaded_model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_strength(mix: ConcreteMix):
    """Predict compressive strength for a single concrete mix"""
    
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        mix_dict = mix.dict()
        df = pd.DataFrame([mix_dict])
        
        # Engineer features
        df_feat = engineer_features(df)
        
        # Ensure correct feature order
        df_feat = df_feat[feature_names]
        
        # Scale features
        if scaler is not None:
            X_scaled = scaler.transform(df_feat)
        else:
            X_scaled = df_feat.values
        
        # Predict
        prediction = loaded_model.predict(X_scaled)[0]
        
        # Get confidence intervals (if Random Forest)
        lower_bound = prediction * 0.90  # Conservative estimate
        upper_bound = prediction * 1.10
        
        # Classify strength
        quality = classify_strength(prediction)
        
        return PredictionResponse(
            predicted_strength=round(float(prediction), 2),
            confidence_interval_lower=round(float(lower_bound), 2),
            confidence_interval_upper=round(float(upper_bound), 2),
            model_used="XGBoost",
            input_features=mix_dict,
            quality_assessment=quality
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict compressive strength for multiple concrete mixes"""
    
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = []
        strengths = []
        
        for mix in request.mixes:
            # Convert to DataFrame
            mix_dict = mix.dict()
            df = pd.DataFrame([mix_dict])
            
            # Engineer features
            df_feat = engineer_features(df)
            df_feat = df_feat[feature_names]
            
            # Scale and predict
            if scaler is not None:
                X_scaled = scaler.transform(df_feat)
            else:
                X_scaled = df_feat.values
            
            prediction = loaded_model.predict(X_scaled)[0]
            strengths.append(prediction)
            
            # Get confidence intervals
            lower_bound = prediction * 0.90
            upper_bound = prediction * 1.10
            
            quality = classify_strength(prediction)
            
            predictions.append(PredictionResponse(
                predicted_strength=round(float(prediction), 2),
                confidence_interval_lower=round(float(lower_bound), 2),
                confidence_interval_upper=round(float(upper_bound), 2),
                model_used="XGBoost",
                input_features=mix_dict,
                quality_assessment=quality
            ))
        
        # Calculate summary statistics
        summary = {
            "total_predictions": len(predictions),
            "average_strength": round(float(np.mean(strengths)), 2),
            "min_strength": round(float(np.min(strengths)), 2),
            "max_strength": round(float(np.max(strengths)), 2),
            "std_strength": round(float(np.std(strengths)), 2)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model"""
    
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_name="XGBoost Regressor",
        version="1.0.0",
        accuracy_metrics=model_metrics,
        feature_names=feature_names
    )

@app.post("/optimize/mix")
async def optimize_mix(target_strength: float, constraints: Optional[Dict] = None):
    """Suggest optimal mix design for target strength (simplified version)"""
    
    # This is a simplified heuristic approach
    # For production, use optimization algorithms (scipy.optimize, genetic algorithms)
    
    suggestions = []
    
    # Rule-based suggestions
    if target_strength < 25:
        suggestions.append({
            "cement": 250,
            "blast_furnace_slag": 50,
            "fly_ash": 0,
            "water": 180,
            "superplasticizer": 3,
            "coarse_aggregate": 1000,
            "fine_aggregate": 750,
            "age": 28,
            "note": "Low-strength mix for non-structural applications"
        })
    elif target_strength < 35:
        suggestions.append({
            "cement": 320,
            "blast_furnace_slag": 80,
            "fly_ash": 0,
            "water": 175,
            "superplasticizer": 5,
            "coarse_aggregate": 1050,
            "fine_aggregate": 700,
            "age": 28,
            "note": "Standard structural concrete mix"
        })
    elif target_strength < 45:
        suggestions.append({
            "cement": 380,
            "blast_furnace_slag": 100,
            "fly_ash": 20,
            "water": 160,
            "superplasticizer": 8,
            "coarse_aggregate": 1000,
            "fine_aggregate": 650,
            "age": 28,
            "note": "High-strength mix for structural applications"
        })
    else:
        suggestions.append({
            "cement": 450,
            "blast_furnace_slag": 120,
            "fly_ash": 30,
            "water": 145,
            "superplasticizer": 12,
            "coarse_aggregate": 950,
            "fine_aggregate": 650,
            "age": 28,
            "note": "Very high-strength mix for special applications"
        })
    
    return {
        "target_strength": target_strength,
        "suggested_mixes": suggestions,
        "disclaimer": "These are suggested starting points. Always validate with lab testing."
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)