import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing utilities for uplift modeling"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load data from CSV file"""
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df, strategy='fill_zero'):
        """Handle missing values in the dataset"""
        print(f"Handling missing values using strategy: {strategy}")
        
        if strategy == 'fill_zero':
            df = df.fillna(0)
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        print(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_features(self, df, categorical_columns=None):
        """Encode categorical features"""
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns
        
        if len(categorical_columns) > 0:
            print(f"Encoding {len(categorical_columns)} categorical features")
            
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
        
        return df
    
    def scale_features(self, df, feature_columns, fit=True):
        """Scale numerical features"""
        if fit:
            df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        else:
            df[feature_columns] = self.scaler.transform(df[feature_columns])
        
        return df
    
    def split_data(self, df, treatment_col, response_col, test_size=0.3, random_state=42):
        """Split data into train and test sets"""
        X = df.drop([treatment_col, response_col], axis=1)
        treatment = df[treatment_col]
        response = df[response_col]
        
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=test_size, random_state=random_state, stratify=treatment
        )
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        
        return X_train, X_test, t_train, t_test, y_train, y_test
    
    def get_feature_importance(self, model, feature_names):
        """Extract feature importance from model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
        else:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def check_data_quality(self, df):
        """Check data quality issues"""
        print("=== Data Quality Check ===")
        
        # Check missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Missing values found: {missing_values.sum()}")
            print(missing_values[missing_values > 0])
        
        # Check duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Duplicate rows: {duplicates}")
        
        # Check data types
        print(f"Data types:")
        print(df.dtypes.value_counts())
        
        # Check for constant features
        constant_features = []
        for col in df.columns:
            if df[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"Constant features: {constant_features}")
        
        return {
            'missing_values': missing_values,
            'duplicates': duplicates,
            'constant_features': constant_features
        }

def preprocess_uplift_data(file_path, treatment_col='treatment_ai_content', response_col='response'):
    """Complete preprocessing pipeline for uplift modeling"""
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data(file_path)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Encode categorical features
    df = preprocessor.encode_categorical_features(df)
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test = preprocessor.split_data(
        df, treatment_col, response_col
    )
    
    # Check data quality
    quality_report = preprocessor.check_data_quality(df)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        't_train': t_train,
        't_test': t_test,
        'y_train': y_train,
        'y_test': y_test,
        'quality_report': quality_report,
        'preprocessor': preprocessor
    }

if __name__ == "__main__":
    # Example usage
    result = preprocess_uplift_data('uplift_model_data.csv')
    print("Preprocessing completed successfully!") 