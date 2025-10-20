import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConcreteDataProcessor:
    """
    ETL Pipeline for Concrete Quality Prediction
    Handles data loading, cleaning, feature engineering, and preprocessing
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'cement', 'blast_furnace_slag', 'fly_ash', 'water',
            'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'
        ]
        self.target_name = 'compressive_strength'
        
    def load_data(self, filepath):
        """Load concrete dataset from CSV or Excel"""
        try:
            if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, df):
        """Clean and validate concrete mix data"""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Handle missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
            # Impute with median for numerical columns
            df_clean = df_clean.fillna(df_clean.median())
        
        # Remove outliers using IQR method
        for col in self.feature_names + [self.target_name]:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"Removing {outliers} outliers from {col}")
                    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        # Validate physical constraints
        df_clean = self._validate_constraints(df_clean)
        
        logger.info(f"Data cleaned: {df_clean.shape}")
        return df_clean
    
    def _validate_constraints(self, df):
        """Ensure concrete mix ratios make physical sense"""
        # Water-cement ratio should be between 0.3 and 1.0
        if 'cement' in df.columns and 'water' in df.columns:
            df = df[(df['water'] / df['cement']).between(0.2, 1.2)]
        
        # All ingredient quantities should be non-negative
        for col in self.feature_names:
            if col in df.columns:
                df = df[df[col] >= 0]
        
        # Age should be positive
        if 'age' in df.columns:
            df = df[df['age'] > 0]
        
        return df
    
    def engineer_features(self, df):
        """Create derived features for better prediction"""
        df_feat = df.copy()
        
        # Water-cement ratio (critical for strength)
        if 'cement' in df_feat.columns and 'water' in df_feat.columns:
            df_feat['water_cement_ratio'] = df_feat['water'] / (df_feat['cement'] + 1e-6)
        
        # Total cementitious materials
        cement_cols = ['cement', 'blast_furnace_slag', 'fly_ash']
        available_cement_cols = [col for col in cement_cols if col in df_feat.columns]
        if available_cement_cols:
            df_feat['total_cementitious'] = df_feat[available_cement_cols].sum(axis=1)
        
        # Total aggregates
        agg_cols = ['coarse_aggregate', 'fine_aggregate']
        available_agg_cols = [col for col in agg_cols if col in df_feat.columns]
        if available_agg_cols:
            df_feat['total_aggregate'] = df_feat[available_agg_cols].sum(axis=1)
        
        # Fine to coarse aggregate ratio
        if 'fine_aggregate' in df_feat.columns and 'coarse_aggregate' in df_feat.columns:
            df_feat['fine_coarse_ratio'] = df_feat['fine_aggregate'] / (df_feat['coarse_aggregate'] + 1e-6)
        
        # Superplasticizer per unit cement
        if 'superplasticizer' in df_feat.columns and 'cement' in df_feat.columns:
            df_feat['sp_cement_ratio'] = df_feat['superplasticizer'] / (df_feat['cement'] + 1e-6)
        
        # Age categories (strength develops over time)
        if 'age' in df_feat.columns:
            df_feat['age_log'] = np.log1p(df_feat['age'])
            df_feat['age_sqrt'] = np.sqrt(df_feat['age'])
            df_feat['is_early_age'] = (df_feat['age'] <= 7).astype(int)
            df_feat['is_mature'] = (df_feat['age'] >= 28).astype(int)
        
        # Replacement percentage
        if all(col in df_feat.columns for col in ['blast_furnace_slag', 'fly_ash', 'total_cementitious']):
            df_feat['replacement_pct'] = (df_feat['blast_furnace_slag'] + df_feat['fly_ash']) / (df_feat['total_cementitious'] + 1e-6)
        
        logger.info(f"Features engineered: {df_feat.shape[1]} total features")
        return df_feat
    
    def preprocess(self, df, fit=True):
        """Scale features for model training"""
        df_processed = df.copy()
        
        # Separate features and target
        if self.target_name in df_processed.columns:
            y = df_processed[self.target_name]
            X = df_processed.drop(columns=[self.target_name])
        else:
            y = None
            X = df_processed
        
        # Scale numerical features
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def run_pipeline(self, filepath, test_size=0.2):
        """Execute complete ETL pipeline"""
        logger.info("Starting ETL pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Engineer features
        df_feat = self.engineer_features(df_clean)
        
        # Preprocess
        X, y = self.preprocess(df_feat, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)
        
        logger.info("ETL pipeline completed successfully!")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'scaler': self.scaler
        }

# Example usage
if __name__ == "__main__":
    processor = ConcreteDataProcessor()
    
    # Download UCI dataset first
    # URL: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength
    
    # Run pipeline
    # data = processor.run_pipeline('data/raw/concrete_data.csv')
    print("ETL Pipeline ready for use!")