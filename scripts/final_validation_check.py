import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def final_validation_check():
    """Final validation check, identifying other issues that may cause high accuracy"""
    print("=== Final Validation Check ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # 1. Check if data distribution is too simple
    print("\n=== 1. Data Distribution Check ===")
    
    # Check response distribution
    response_dist = df['response'].value_counts(normalize=True)
    print(f"Response distribution:")
    for value, ratio in response_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Check treatment distribution
    treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
    print(f"\nTreatment distribution:")
    for value, ratio in treatment_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # Check if data is too imbalanced
    if response_dist.min() < 0.01:  # If any category accounts for less than 1%
        print("⚠️  Data severely imbalanced, may cause model overfitting")
    
    # 2. Check if features are too simple
    print("\n=== 2. Feature Complexity Check ===")
    
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    # Check unique value count of features
    simple_features = []
    for col in numeric_features:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.01:  # If unique value ratio is less than 1%
            simple_features.append((col, unique_ratio))
    
    print(f"Overly simple features (unique value ratio < 1%):")
    for col, ratio in simple_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 3. Check for deterministic relationships
    print("\n=== 3. Deterministic Relationship Check ===")
    
    # Check if any features have perfect correlation with response
    perfect_corr_features = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['response']))
        if corr > 0.95:  # If correlation exceeds 95%
            perfect_corr_features.append((col, corr))
    
    print(f"Features almost perfectly correlated with response (>95%):")
    for col, corr in perfect_corr_features:
        print(f"  {col}: {corr:.4f}")
    
    # 4. Check model complexity
    print("\n=== 4. Model Complexity Check ===")
    
    # Test with different model complexities
    model_configs = [
        {'name': 'Simple', 'n_estimators': 10, 'max_depth': 2},
        {'name': 'Medium', 'n_estimators': 50, 'max_depth': 4},
        {'name': 'Complex', 'n_estimators': 100, 'max_depth': 8},
        {'name': 'Very Complex', 'n_estimators': 200, 'max_depth': 12}
    ]
    
    X = df[numeric_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.3, random_state=42, stratify=treatment
    )
    
    complexity_results = []
    
    for config in model_configs:
        print(f"\nTesting {config['name']} model:")
        
        # Train models
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
        treatment_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            random_state=42, verbosity=0
        )
        control_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            random_state=42, verbosity=0
        )
        
        treatment_model.fit(X_treatment.values, y_treatment.values)
        control_model.fit(X_control.values, y_control.values)
        
        # Predict
        y_pred_treatment = treatment_model.predict(X_test.values)
        y_pred_control = control_model.predict(X_test.values)
        
        # Calculate uplift
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        complexity_results.append({
            'name': config['name'],
            'uplift_accuracy': uplift_accuracy,
            'actual_uplift': actual_uplift,
            'predicted_uplift': uplift_pred
        })
        
        print(f"  Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # Check impact of complexity on accuracy
    print(f"\nImpact of model complexity on accuracy:")
    for result in complexity_results:
        print(f"  {result['name']}: {result['uplift_accuracy']:.2%}")
    
    # 5. Check other forms of data leakage
    print("\n=== 5. Other Data Leakage Checks ===")
    
    # Check if any features directly equal response
    direct_response_features = []
    for col in numeric_features:
        if df[col].equals(df['response']):
            direct_response_features.append(col)
    
    print(f"Features directly equal to response: {direct_response_features}")
    
    # Check if any features are linear transformations of response
    linear_response_features = []
    for col in numeric_features:
        if col != 'response':
            # Check if linearly related to response
            corr = abs(df[col].corr(df['response']))
            if corr > 0.99:  # If correlation exceeds 99%
                linear_response_features.append((col, corr))
    
    print(f"Features almost linearly related to response (>99%):")
    for col, corr in linear_response_features:
        print(f"  {col}: {corr:.4f}")
    
    # 6. Check feature selection issues
    print("\n=== 6. Feature Selection Issue Check ===")
    
    # Check for duplicate features
    duplicate_features = []
    for i, col1 in enumerate(numeric_features):
        for j, col2 in enumerate(numeric_features[i+1:], i+1):
            if df[col1].equals(df[col2]):
                duplicate_features.append((col1, col2))
    
    print(f"Completely duplicate feature pairs:")
    for col1, col2 in duplicate_features:
        print(f"  {col1} = {col2}")
    
    # Check for highly correlated features
    high_corr_pairs = []
    for i, col1 in enumerate(numeric_features):
        for j, col2 in enumerate(numeric_features[i+1:], i+1):
            corr = abs(df[col1].corr(df[col2]))
            if corr > 0.95:
                high_corr_pairs.append((col1, col2, corr))
    
    print(f"Highly correlated feature pairs (>95%):")
    for col1, col2, corr in high_corr_pairs:
        print(f"  {col1} <-> {col2}: {corr:.4f}")
    
    # 7. Check data quality issues
    print("\n=== 7. Data Quality Issue Check ===")
    
    # Check missing values
    missing_info = df[numeric_features].isnull().sum()
    high_missing = missing_info[missing_info > 0]
    print(f"Features with missing values:")
    for col, missing in high_missing.items():
        print(f"  {col}: {missing:,} ({missing/len(df):.2%})")
    
    # Check outliers
    outlier_features = []
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_ratio = outliers / len(df)
        if outlier_ratio > 0.2:  # If outliers exceed 20%
            outlier_features.append((col, outlier_ratio))
    
    print(f"\nFeatures with high outlier ratio (>20%):")
    for col, ratio in outlier_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 8. Check randomness
    print("\n=== 8. Randomness Check ===")
    
    # Test with different random seeds
    seeds = [42, 123, 456, 789, 999]
    random_results = []
    
    for seed in seeds:
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=0.3, random_state=seed, stratify=treatment
        )
        
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
        treatment_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=seed, verbosity=0)
        control_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=seed, verbosity=0)
        
        treatment_model.fit(X_treatment.values, y_treatment.values)
        control_model.fit(X_control.values, y_control.values)
        
        y_pred_treatment = treatment_model.predict(X_test.values)
        y_pred_control = control_model.predict(X_test.values)
        
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        random_results.append(uplift_accuracy)
    
    print(f"Accuracy for different random seeds:")
    for i, accuracy in enumerate(random_results):
        print(f"  Seed {seeds[i]}: {accuracy:.2%}")
    
    accuracy_variance = np.var(random_results)
    print(f"Accuracy variance: {accuracy_variance:.4f}")
    
    if accuracy_variance < 0.001:
        print("⚠️  Accuracy is too stable, may indicate deterministic relationships")
    
    # 9. Final Conclusion
    print("\n=== 9. Final Conclusion ===")
    
    issues = []
    
    if len(perfect_corr_features) > 0:
        issues.append("Found features almost perfectly correlated with response")
    
    if len(direct_response_features) > 0:
        issues.append("Found features directly equal to response")
    
    if len(linear_response_features) > 0:
        issues.append("Found features almost linearly related to response")
    
    if len(duplicate_features) > 0:
        issues.append("Found duplicate features")
    
    if accuracy_variance < 0.001:
        issues.append("Accuracy is too stable, may indicate deterministic relationships")
    
    if issues:
        print("⚠️  Found the following issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nSuggestions:")
        print("  1. Remove features highly correlated with response")
        print("  2. Remove duplicate features")
        print("  3. Check data preprocessing steps")
        print("  4. Re-design feature engineering")
    else:
        print("✅ No obvious deterministic relationship issues found")
    
    return {
        'perfect_corr_features': perfect_corr_features,
        'direct_response_features': direct_response_features,
        'linear_response_features': linear_response_features,
        'duplicate_features': duplicate_features,
        'high_corr_pairs': high_corr_pairs,
        'outlier_features': outlier_features,
        'accuracy_variance': accuracy_variance,
        'issues': issues
    }

if __name__ == "__main__":
    results = final_validation_check() 