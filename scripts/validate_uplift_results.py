import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def validate_uplift_results():
    """Strict validation of uplift modeling results"""
    print("=== Strict Uplift Modeling Validation ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data distribution
    print("\n=== Data Distribution Check ===")
    print(f"Treatment distribution:")
    print(f"  Control (0): {(df['treatment_ai_content'] == 0).sum():,} ({(df['treatment_ai_content'] == 0).mean():.2%})")
    print(f"  Treatment (1): {(df['treatment_ai_content'] == 1).sum():,} ({(df['treatment_ai_content'] == 1).mean():.2%})")
    
    print(f"\nResponse distribution:")
    print(f"  No click (0): {(df['response'] == 0).sum():,} ({(df['response'] == 0).mean():.2%})")
    print(f"  Click (1): {(df['response'] == 1).sum():,} ({(df['response'] == 1).mean():.2%})")
    
    # Check correlation between features and treatment
    print("\n=== Feature-Treatment Correlation Check ===")
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    
    treatment_correlations = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            corr = abs(df[col].corr(df['treatment_ai_content']))
            treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 features with highest correlation to treatment:")
    for col, corr in treatment_correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # Check data leakage
    print("\n=== Data Leakage Check ===")
    high_corr_features = [col for col, corr in treatment_correlations if corr > 0.5]
    print(f"Number of features with correlation > 0.5 to treatment: {len(high_corr_features)}")
    if high_corr_features:
        print("⚠️  Warning: Potential data leakage!")
        for col in high_corr_features:
            print(f"  - {col}")
    
    # Use stricter validation methods
    print("\n=== Strict Validation Methods ===")
    
    # 1. Larger test set
    X = df[feature_cols].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # Use 50% test set
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.5, random_state=42, stratify=treatment
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # 2. Calculate actual uplift
    actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print(f"\nActual Uplift: {actual_uplift:.4f}")
    
    # 3. Train models and predict
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    # Train models
    treatment_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=4, subsample=0.7,
        learning_rate=0.1, random_state=42, verbosity=0
    )
    control_model = xgb.XGBRegressor(
        n_estimators=50, max_depth=4, subsample=0.7,
        learning_rate=0.1, random_state=42, verbosity=0
    )
    
    treatment_model.fit(X_treatment.values, y_treatment.values)
    control_model.fit(X_control.values, y_control.values)
    
    # Predict
    y_pred_treatment = treatment_model.predict(X_test.values)
    y_pred_control = control_model.predict(X_test.values)
    
    # Calculate uplift prediction
    uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
    uplift_error = abs(actual_uplift - uplift_pred)
    uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
    
    print(f"\n=== Strict Validation Results ===")
    print(f"Actual Uplift: {actual_uplift:.4f}")
    print(f"Predicted Uplift: {uplift_pred:.4f}")
    print(f"Uplift Error: {uplift_error:.4f}")
    print(f"Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # 4. Cross validation
    print(f"\n=== Cross Validation Results ===")
    cv_scores_treatment = cross_val_score(treatment_model, X_treatment.values, y_treatment.values, cv=5, scoring='r2')
    cv_scores_control = cross_val_score(control_model, X_control.values, y_control.values, cv=5, scoring='r2')
    
    print(f"Treatment Model CV R²: {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
    print(f"Control Model CV R²: {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
    
    # 5. Check prediction distribution
    print(f"\n=== Prediction Distribution Check ===")
    print(f"Treatment prediction mean: {y_pred_treatment.mean():.4f}")
    print(f"Control prediction mean: {y_pred_control.mean():.4f}")
    print(f"Treatment prediction variance: {y_pred_treatment.var():.4f}")
    print(f"Control prediction variance: {y_pred_control.var():.4f}")
    
    # 6. Check if prediction values are in reasonable range
    print(f"\n=== Prediction Value Range Check ===")
    print(f"Treatment prediction range: [{y_pred_treatment.min():.4f}, {y_pred_treatment.max():.4f}]")
    print(f"Control prediction range: [{y_pred_control.min():.4f}, {y_pred_control.max():.4f}]")
    
    # Check if all prediction values are in [0,1] range
    treatment_in_range = np.all((y_pred_treatment >= 0) & (y_pred_treatment <= 1))
    control_in_range = np.all((y_pred_control >= 0) & (y_pred_control <= 1))
    
    print(f"Treatment predictions in [0,1] range: {treatment_in_range}")
    print(f"Control predictions in [0,1] range: {control_in_range}")
    
    # 7. Conclusion
    print(f"\n=== Validation Conclusion ===")
    if uplift_accuracy > 0.95:
        print("⚠️  Warning: Accuracy too high, potential issues:")
        print("  1. Data leakage - Features contain treatment information")
        print("  2. Overfitting - Model too complex")
        print("  3. Test set too small - Unstable results")
        print("  4. Feature engineering issues - Some features directly predict target")
    else:
        print("✅ Reasonable accuracy, model is credible")
    
    if len(high_corr_features) > 0:
        print("⚠️  Found features highly correlated with treatment, potential data leakage")
    
    return {
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_pred,
        'uplift_error': uplift_error,
        'uplift_accuracy': uplift_accuracy,
        'high_corr_features': high_corr_features
    }

if __name__ == "__main__":
    results = validate_uplift_results() 