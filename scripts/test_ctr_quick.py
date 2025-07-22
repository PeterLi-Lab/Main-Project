import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def test_ctr_quick():
    """Quick CTR model test"""
    print("=== Quick CTR Model Test ===\n")
    
    # Load data
    df = pd.read_csv('user_post_click_samples.csv')
    print(f"Total data volume: {len(df):,}")
    
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
    
    # Check feature types
    numeric_features = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
    
    print(f"\nNumeric features: {len(numeric_features)}")
    
    # Handle missing values
    df = df.fillna(0)
    
    # Prepare data for training
    X = df[numeric_features]
    y = df[target_col]
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Valid samples: {len(X):,}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train quick model
    print(f"\n=== Training Quick Model ===")
    
    model = xgb.XGBClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    
    # Train model
    model.fit(X_train.values, y_train.values)
    
    # Predict
    y_pred = model.predict(X_test.values)
    y_pred_proba = model.predict_proba(X_test.values)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Quick model performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Quick assessment
    print(f"\n=== Quick Assessment ===")
    
    if accuracy > 0.6:
        print("✅ Model accuracy is reasonable")
    else:
        print("❌ Model accuracy is poor")
    
    if auc > 0.6:
        print("✅ Model AUC is reasonable")
    else:
        print("❌ Model AUC is poor")
    
    return {
        'accuracy': accuracy,
        'auc': auc
    }

if __name__ == "__main__":
    results = test_ctr_quick() 