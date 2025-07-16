import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def clean_uplift_analysis_fixed():
    """Clean uplift analysis with problematic features removed"""
    print("=== Clean Uplift Analysis (Fixed Version) ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data size: {len(df):,}")
    
    # Define features to remove (leaky features)
    leaky_features = [
        'ai_interest_x_treatment',      # Direct treatment information
        'user_ai_interest_score',       # Highly correlated with treatment
        'user_previous_ai_click_rate',  # Highly correlated with treatment
        'user_ai_interest_weighted',    # Highly correlated with treatment
        'user_ai_interactions'          # Highly correlated with treatment
    ]
    
    # Define redundant features to remove
    redundant_features = [
        'user_previous_ai_click_rate',  # Duplicate of user_ai_interest_score
        'user_ai_interest_weighted',    # Highly correlated
        'total_votes',                  # Highly correlated with Score
        'upvotes',                      # Highly correlated with Score
        'user_post_tag_overlap'         # Duplicate of num_tags
    ]
    
    # Get clean features
    all_features = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    clean_features = [col for col in all_features if col not in leaky_features + redundant_features]
    
    # Only use numeric features
    clean_features = [col for col in clean_features if df[col].dtype in ['int64', 'float64']]
    
    print(f"Clean features count: {len(clean_features)}")
    print("Clean features list:")
    for feature in clean_features:
        print(f"  - {feature}")
    
    # Check data quality
    valid_samples = len(df.dropna(subset=clean_features))
    print(f"\nValid samples: {valid_samples:,}")
    
    # Check correlation with treatment for clean features
    print("\n=== Clean Features vs Treatment Correlation Check ===")
    treatment_correlations = []
    for col in clean_features:
        if df[col].dtype in ['int64', 'float64']:
            corr = abs(df[col].corr(df['treatment_ai_content']))
            treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 clean features with highest treatment correlation:")
    for col, corr in treatment_correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # Check for high correlation features
    high_corr_features = [col for col, corr in treatment_correlations if corr > 0.3]
    print(f"\nClean features with treatment correlation > 0.3: {len(high_corr_features)}")
    if high_corr_features:
        print("⚠️  Warning: Still have high correlation features!")
        for col in high_corr_features:
            print(f"  - {col}")
    
    # Prepare data for modeling
    X = df[clean_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # Split data
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.5, random_state=42, stratify=treatment
    )
    
    print(f"\nTraining set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Calculate actual uplift
    actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print(f"\nActual Uplift: {actual_uplift:.4f}")
    
    # Split training data by treatment
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    print(f"\nTraining treatment samples: {len(X_treatment):,}")
    print(f"Training control samples: {len(X_control):,}")
    
    # Train XGBoost models
    print("\n=== Training XGBoost Models ===")
    
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
    
    # Make predictions
    y_pred_treatment = treatment_model.predict(X_test.values)
    y_pred_control = control_model.predict(X_test.values)
    
    # Calculate uplift prediction
    uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
    uplift_error = abs(actual_uplift - uplift_pred)
    uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
    
    print(f"\n=== Clean Features Results ===")
    print(f"Actual Uplift: {actual_uplift:.4f}")
    print(f"Predicted Uplift: {uplift_pred:.4f}")
    print(f"Uplift Error: {uplift_error:.4f}")
    print(f"Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # Cross-validation
    cv_scores_treatment = cross_val_score(treatment_model, X_treatment.values, y_treatment.values, cv=5, scoring='r2')
    cv_scores_control = cross_val_score(control_model, X_control.values, y_control.values, cv=5, scoring='r2')
    
    print(f"\nCross-validation results:")
    print(f"Treatment Model CV R²: {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
    print(f"Control Model CV R²: {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
    
    # Feature importance
    print(f"\n=== Feature Importance (Treatment Model) ===")
    treatment_importance = treatment_model.feature_importances_
    feature_importance = list(zip(clean_features, treatment_importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 feature importance:")
    for feature, importance in feature_importance[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # Prediction distribution check
    print(f"\n=== Prediction Distribution Check ===")
    print(f"Treatment prediction mean: {y_pred_treatment.mean():.4f}")
    print(f"Control prediction mean: {y_pred_control.mean():.4f}")
    print(f"Treatment prediction variance: {y_pred_treatment.var():.4f}")
    print(f"Control prediction variance: {y_pred_control.var():.4f}")
    
    # Check prediction range
    treatment_in_range = np.all((y_pred_treatment >= 0) & (y_pred_treatment <= 1))
    control_in_range = np.all((y_pred_control >= 0) & (y_pred_control <= 1))
    
    print(f"Treatment predictions in [0,1] range: {treatment_in_range}")
    print(f"Control predictions in [0,1] range: {control_in_range}")
    
    # Conclusion
    print(f"\n=== Conclusion ===")
    if uplift_accuracy > 0.95:
        print("⚠️  Warning: Still high accuracy, may need further investigation")
    else:
        print("✅ Clean feature set achieved reasonable accuracy")
    
    if actual_uplift * uplift_pred > 0:
        print("✅ Uplift prediction direction: Correct")
    else:
        print("❌ Uplift prediction direction: Incorrect")
    
    return {
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_pred,
        'uplift_error': uplift_error,
        'uplift_accuracy': uplift_accuracy,
        'clean_features': clean_features,
        'high_corr_features': high_corr_features
    }

if __name__ == "__main__":
    results = clean_uplift_analysis_fixed() 