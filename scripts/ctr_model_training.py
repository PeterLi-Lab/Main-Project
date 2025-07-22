import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def ctr_model_training():
    """CTR (Click-Through Rate) model training"""
    print("=== CTR Model Training ===\n")
    
    # Load data
    df = pd.read_csv('user_post_click_samples.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check target variable distribution
    if 'click' in df.columns:
        target_col = 'click'
    elif 'response' in df.columns:
        target_col = 'response'
    else:
        print("❌ No target variable found (click or response)")
        return None
    
    print(f"\nTarget variable: {target_col}")
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
    X = df[feature_cols]
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
    
    # Train different models
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, verbosity=0
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        
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
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train.values, y_train.values, cv=5, scoring='accuracy')
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Compare models
    print(f"\n=== Model Comparison ===")
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1 Score': [results[model]['f1'] for model in results.keys()],
        'AUC': [results[model]['auc'] for model in results.keys()],
        'CV Accuracy': [results[model]['cv_mean'] for model in results.keys()]
    })
    
    print(comparison_df.to_string(index=False))
    
    # Feature importance for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    print(f"\n=== Feature Importance ({best_model_name}) ===")
    
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("Top 15 most important features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save results
    print(f"\n=== Saving Results ===")
    
    # Save best model
    import joblib
    joblib.dump(best_model, f'models/best_ctr_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    print(f"Best model saved: models/best_ctr_model_{best_model_name.lower().replace(' ', '_')}.pkl")
    
    # Save feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance_df.to_csv('output/ctr_feature_importance.csv', index=False)
        print("Feature importance saved: output/ctr_feature_importance.csv")
    
    # Save comparison results
    comparison_df.to_csv('output/ctr_model_comparison.csv', index=False)
    print("Model comparison saved: output/ctr_model_comparison.csv")
    
    return {
        'results': results,
        'best_model': best_model_name,
        'feature_importance': importance_df if hasattr(best_model, 'feature_importances_') else None,
        'comparison': comparison_df
    }

if __name__ == "__main__":
    results = ctr_model_training() 