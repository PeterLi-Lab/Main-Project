import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def final_clean_uplift_analysis_english():
    """Final clean uplift analysis with all problematic features removed"""
    print("=== Final Clean Uplift Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data size: {len(df):,}")
    
    # 1. Remove all problematic features
    print("\n=== 1. Feature Cleaning ===")
    
    # Features to remove due to data leakage
    leaky_features = [
        'ai_interest_x_treatment',      # Direct treatment information
        'user_ai_interest_score',       # Highly correlated with treatment
        'user_previous_ai_click_rate',  # Highly correlated with treatment
        'user_ai_interest_weighted',    # Highly correlated with treatment
        'user_ai_interactions'          # Highly correlated with treatment
    ]
    
    # Features to remove due to high correlation
    duplicate_features = [
        'user_previous_ai_click_rate',  # Duplicate of user_ai_interest_score
        'user_ai_interest_weighted',    # Highly correlated
        'total_votes',                  # Highly correlated with Score
        'upvotes',                      # Highly correlated with Score
        'user_post_tag_overlap'         # Duplicate of num_tags
    ]
    
    # Get all features to remove
    all_features_to_remove = list(set(leaky_features + duplicate_features))
    
    # Get remaining features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    clean_features = [col for col in feature_cols if col not in all_features_to_remove]
    
    # Filter only numeric features
    numeric_features = [col for col in clean_features if df[col].dtype in ['int64', 'float64']]
    
    print(f"Original features: {len(feature_cols)}")
    print(f"Features removed: {len(all_features_to_remove)}")
    print(f"Clean numeric features: {len(numeric_features)}")
    
    print(f"\nRemoved features:")
    for feature in all_features_to_remove:
        print(f"  - {feature}")
    
    print(f"\nClean features:")
    for feature in numeric_features:
        print(f"  - {feature}")
    
    # 2. Data preparation
    print("\n=== 2. Data Preparation ===")
    
    X = df[numeric_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Treatment distribution: {treatment.value_counts(normalize=True).to_dict()}")
    print(f"Response distribution: {response.value_counts(normalize=True).to_dict()}")
    
    # 3. Check correlations after cleaning
    print("\n=== 3. Correlation Check After Cleaning ===")
    
    # Check treatment correlations
    treatment_correlations = []
    for col in numeric_features:
        corr = abs(X[col].corr(treatment))
        treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Top 10 features with highest treatment correlation:")
    for i, (col, corr) in enumerate(treatment_correlations[:10]):
        print(f"{i+1}. {col}: {corr:.4f}")
    
    # Check response correlations
    response_correlations = []
    for col in numeric_features:
        corr = abs(X[col].corr(response))
        response_correlations.append((col, corr))
    
    response_correlations.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 features with highest response correlation:")
    for i, (col, corr) in enumerate(response_correlations[:10]):
        print(f"{i+1}. {col}: {corr:.4f}")
    
    # 4. Train models
    print("\n=== 4. Model Training ===")
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.3, random_state=42, stratify=treatment
    )
    
    print(f"Training set size: {len(X_train):,}")
    print(f"Test set size: {len(X_test):,}")
    
    # Train treatment and control models
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    print(f"Treatment group training samples: {len(X_treatment):,}")
    print(f"Control group training samples: {len(X_control):,}")
    
    # Train models
    treatment_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
    control_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
    
    treatment_model.fit(X_treatment.values, y_treatment.values)
    control_model.fit(X_control.values, y_control.values)
    
    # 5. Make predictions
    print("\n=== 5. Model Predictions ===")
    
    y_pred_treatment = treatment_model.predict(X_test.values)
    y_pred_control = control_model.predict(X_test.values)
    
    # Calculate actual and predicted uplift
    treatment_mask_test = t_test == 1
    control_mask_test = t_test == 0
    
    actual_uplift = y_test[treatment_mask_test].mean() - y_test[control_mask_test].mean()
    uplift_pred = y_pred_treatment[treatment_mask_test].mean() - y_pred_control[control_mask_test].mean()
    
    print(f"Actual uplift: {actual_uplift:.4f}")
    print(f"Predicted uplift: {uplift_pred:.4f}")
    print(f"Uplift error: {abs(actual_uplift - uplift_pred):.4f}")
    
    # Calculate accuracy
    uplift_error = abs(actual_uplift - uplift_pred)
    uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
    
    print(f"Uplift accuracy: {uplift_accuracy:.2%}")
    
    # 6. Cross-validation
    print("\n=== 6. Cross-Validation ===")
    
    # Cross-validation for treatment model
    treatment_cv_scores = cross_val_score(
        treatment_model, X_treatment.values, y_treatment.values, 
        cv=5, scoring='r2'
    )
    
    # Cross-validation for control model
    control_cv_scores = cross_val_score(
        control_model, X_control.values, y_control.values, 
        cv=5, scoring='r2'
    )
    
    print(f"Treatment model CV R²: {treatment_cv_scores.mean():.4f} ± {treatment_cv_scores.std():.4f}")
    print(f"Control model CV R²: {control_cv_scores.mean():.4f} ± {control_cv_scores.std():.4f}")
    
    # 7. Feature importance
    print("\n=== 7. Feature Importance ===")
    
    # Treatment model feature importance
    treatment_importance = treatment_model.feature_importances_
    treatment_feature_importance = list(zip(numeric_features, treatment_importance))
    treatment_feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Treatment model top 10 important features:")
    for i, (feature, importance) in enumerate(treatment_feature_importance[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Control model feature importance
    control_importance = control_model.feature_importances_
    control_feature_importance = list(zip(numeric_features, control_importance))
    control_feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nControl model top 10 important features:")
    for i, (feature, importance) in enumerate(control_feature_importance[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # 8. Model performance metrics
    print("\n=== 8. Model Performance Metrics ===")
    
    # Treatment model performance
    treatment_r2 = r2_score(y_test[treatment_mask_test], y_pred_treatment[treatment_mask_test])
    treatment_mae = mean_absolute_error(y_test[treatment_mask_test], y_pred_treatment[treatment_mask_test])
    
    # Control model performance
    control_r2 = r2_score(y_test[control_mask_test], y_pred_control[control_mask_test])
    control_mae = mean_absolute_error(y_test[control_mask_test], y_pred_control[control_mask_test])
    
    print(f"Treatment model performance:")
    print(f"  R²: {treatment_r2:.4f}")
    print(f"  MAE: {treatment_mae:.4f}")
    
    print(f"Control model performance:")
    print(f"  R²: {control_r2:.4f}")
    print(f"  MAE: {control_mae:.4f}")
    
    # 9. Final validation
    print("\n=== 9. Final Validation ===")
    
    # Check if accuracy is reasonable
    if uplift_accuracy > 0.95:
        print("⚠️  Accuracy still very high, possible remaining issues")
    elif uplift_accuracy > 0.8:
        print("✅ Accuracy at reasonable level")
    else:
        print("✅ Accuracy at expected level")
    
    # Check uplift direction
    if (actual_uplift > 0 and uplift_pred > 0) or (actual_uplift < 0 and uplift_pred < 0):
        print("✅ Uplift direction correctly predicted")
    else:
        print("⚠️  Uplift direction incorrectly predicted")
    
    # Check feature correlations
    high_corr_treatment = [col for col, corr in treatment_correlations if corr > 0.3]
    if len(high_corr_treatment) > 0:
        print(f"⚠️  Found {len(high_corr_treatment)} features with moderate correlation with treatment")
        for col, corr in treatment_correlations[:5]:
            print(f"  - {col}: {corr:.4f}")
    else:
        print("✅ No features highly correlated with treatment")
    
    # 10. Summary and recommendations
    print("\n=== 10. Summary and Recommendations ===")
    
    print(f"Final results:")
    print(f"  - Clean features used: {len(numeric_features)}")
    print(f"  - Uplift accuracy: {uplift_accuracy:.2%}")
    print(f"  - Treatment model R²: {treatment_r2:.4f}")
    print(f"  - Control model R²: {control_r2:.4f}")
    print(f"  - Actual uplift: {actual_uplift:.4f}")
    print(f"  - Predicted uplift: {uplift_pred:.4f}")
    
    if uplift_accuracy < 0.9:
        print("\n✅ Results look reasonable after cleaning")
        print("Recommendations:")
        print("  1. Use this clean feature set for production")
        print("  2. Monitor model performance on new data")
        print("  3. Consider business interpretation of results")
    else:
        print("\n⚠️  Results still suspicious, further investigation needed")
        print("Recommendations:")
        print("  1. Investigate remaining high correlations")
        print("  2. Check for additional data leakage")
        print("  3. Consider simpler models")
    
    return {
        'clean_features': numeric_features,
        'uplift_accuracy': uplift_accuracy,
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_pred,
        'treatment_r2': treatment_r2,
        'control_r2': control_r2,
        'treatment_mae': treatment_mae,
        'control_mae': control_mae
    }

if __name__ == "__main__":
    results = final_clean_uplift_analysis_english() 