import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def test_simple_ctr():
    """Simple CTR model test"""
    print("=== Simple CTR Model Test ===\n")
    
    # Load data
    df = pd.read_csv('user_post_click_samples.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check target variable
    if 'click' in df.columns:
        target_col = 'click'
    elif 'response' in df.columns:
        target_col = 'response'
    else:
        print("❌ No target variable found (click or response)")
        return None
    
    print(f"Target variable: {target_col}")
    print(f"Target distribution:")
    target_dist = df[target_col].value_counts(normalize=True)
    for value, ratio in target_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in [target_col, 'user_id', 'post_id']]
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Check feature types
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Handle missing values
    print(f"\nHandling missing values...")
    df = df.fillna(0)
    
    # Prepare data for training
    X = df[numeric_features]
    y = df[target_col]
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Valid samples after cleaning: {len(X):,}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train simple model
    print(f"\n=== Training Simple Model ===")
    
    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    
    # Train model
    model.fit(X_train.values, y_train.values)
    
    # Predict
    y_pred = model.predict(X_test.values)
    y_pred_proba = model.predict_proba(X_test.values)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 feature importance:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Quick analysis
    print(f"\n=== Quick Analysis ===")
    
    # Check if model is reasonable
    if accuracy > 0.7:
        print("✅ Model accuracy is good")
    elif accuracy > 0.5:
        print("⚠️  Model accuracy is moderate")
    else:
        print("❌ Model accuracy is poor")
    
    if auc > 0.7:
        print("✅ Model AUC is good")
    elif auc > 0.6:
        print("⚠️  Model AUC is moderate")
    else:
        print("❌ Model AUC is poor")
    
    # Check for overfitting
    train_pred = model.predict(X_train.values)
    train_accuracy = accuracy_score(y_train, train_pred)
    
    if train_accuracy - accuracy > 0.1:
        print("⚠️  Potential overfitting detected")
    else:
        print("✅ No obvious overfitting")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'train_accuracy': train_accuracy,
        'feature_importance': importance_df if hasattr(model, 'feature_importances_') else None
    }

if __name__ == "__main__":
    results = test_simple_ctr() 