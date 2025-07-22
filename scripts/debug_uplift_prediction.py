import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def debug_uplift_prediction():
    """Debug uplift prediction model performance and issues"""
    print("=== Uplift Prediction Debug ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Check for required columns
    required_cols = ['treatment_ai_content', 'response']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return None
    
    print("✅ All required columns present")
    
    # 1. Check treatment and response distributions
    print("\n=== 1. Treatment and Response Analysis ===")
    
    # Treatment distribution
    treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
    print(f"Treatment distribution:")
    for value, ratio in treatment_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Response distribution
    response_dist = df['response'].value_counts(normalize=True)
    print(f"\nResponse distribution:")
    for value, ratio in response_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Treatment-Response relationship
    print(f"\nTreatment-Response relationship:")
    treatment_response = df.groupby('treatment_ai_content')['response'].agg(['mean', 'count'])
    print(treatment_response)
    
    # Calculate uplift
    treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
    control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
    uplift = treatment_response_rate - control_response_rate
    
    print(f"\nUplift analysis:")
    print(f"  Treatment response rate: {treatment_response_rate:.2%}")
    print(f"  Control response rate: {control_response_rate:.2%}")
    print(f"  Uplift: {uplift:.2%}")
    
    # 2. Prepare features
    print("\n=== 2. Feature Preparation ===")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    
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
    df = df.fillna(0)
    
    # 3. Train different models
    print("\n=== 3. Model Training and Comparison ===")
    
    # Prepare data
    X = df[numeric_features]
    y = df['response']
    t = df['treatment_ai_content']
    
    # Remove rows with NaN in target
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    t = t[valid_mask]
    
    # Split data
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=y
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
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
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
    
    # 4. Analyze prediction errors
    print("\n=== 4. Prediction Error Analysis ===")
    
    # Use best model for error analysis
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    print(f"Using {best_model_name} for error analysis")
    
    # Get predictions
    y_pred = best_model.predict(X_test.values)
    y_pred_proba = best_model.predict_proba(X_test.values)[:, 1]
    
    # Analyze errors
    errors = (y_test != y_pred)
    error_rate = errors.mean()
    
    print(f"Overall error rate: {error_rate:.2%}")
    
    # Error analysis by treatment group
    treatment_errors = pd.DataFrame({
        'treatment': t_test,
        'actual': y_test,
        'predicted': y_pred,
        'error': errors
    })
    
    print(f"\nError analysis by treatment group:")
    error_by_treatment = treatment_errors.groupby('treatment')['error'].agg(['mean', 'count'])
    print(error_by_treatment)
    
    # Error analysis by response
    print(f"\nError analysis by response:")
    error_by_response = treatment_errors.groupby('actual')['error'].agg(['mean', 'count'])
    print(error_by_response)
    
    # 5. Check for prediction bias
    print("\n=== 5. Prediction Bias Analysis ===")
    
    # Check if model predictions are biased by treatment
    treatment_prediction = pd.DataFrame({
        'treatment': t_test,
        'predicted_proba': y_pred_proba
    })
    
    treatment_pred_analysis = treatment_prediction.groupby('treatment')['predicted_proba'].agg(['mean', 'std'])
    print(f"Prediction probability by treatment:")
    print(treatment_pred_analysis)
    
    # Check for calibration issues
    print(f"\nCalibration analysis:")
    for treatment in [0, 1]:
        mask = t_test == treatment
        if mask.sum() > 0:
            actual_rate = y_test[mask].mean()
            predicted_rate = y_pred_proba[mask].mean()
            print(f"  Treatment {treatment}: Actual={actual_rate:.2%}, Predicted={predicted_rate:.2%}")
    
    # 6. Analyze feature importance
    print("\n=== 6. Feature Importance Analysis ===")
    
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"Top 15 most important features:")
        for idx, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Check for suspicious features
        suspicious_features = []
        for idx, row in importance_df.head(10).iterrows():
            feature = row['feature']
            # Check if feature name suggests it might be a leak
            if any(keyword in feature.lower() for keyword in ['treatment', 'response', 'target', 'label']):
                suspicious_features.append(feature)
        
        if suspicious_features:
            print(f"\n⚠️  Suspicious features with high importance:")
            for feature in suspicious_features:
                importance_val = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
                print(f"  {feature}: {importance_val:.4f}")
    
    # 7. Summary and recommendations
    print("\n=== 7. Summary and Recommendations ===")
    
    issues = []
    
    if error_rate > 0.3:
        issues.append(f"High error rate: {error_rate:.1%}")
    
    # Check for treatment bias in predictions
    treatment_0_pred = treatment_prediction[treatment_prediction['treatment'] == 0]['predicted_proba'].mean()
    treatment_1_pred = treatment_prediction[treatment_prediction['treatment'] == 1]['predicted_proba'].mean()
    
    if abs(treatment_1_pred - treatment_0_pred) > 0.1:
        issues.append("Model predictions show treatment bias")
    
    if len(suspicious_features) > 0:
        issues.append(f"Found {len(suspicious_features)} suspicious features")
    
    if issues:
        print("⚠️  Found the following prediction issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        if error_rate > 0.3:
            print("  1. Improve feature engineering")
            print("  2. Try different model architectures")
            print("  3. Check for data quality issues")
        if abs(treatment_1_pred - treatment_0_pred) > 0.1:
            print("  4. Address treatment bias in predictions")
        if len(suspicious_features) > 0:
            print("  5. Review suspicious features for data leakage")
    else:
        print("✅ No obvious prediction issues found")
    
    return {
        'results': results,
        'best_model': best_model_name,
        'error_rate': error_rate,
        'treatment_bias': abs(treatment_1_pred - treatment_0_pred),
        'suspicious_features': suspicious_features,
        'issues': issues
    }

if __name__ == "__main__":
    results = debug_uplift_prediction() 