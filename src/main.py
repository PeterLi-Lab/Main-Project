import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function for the project"""
    print("=== Main Project Execution ===\n")
    
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
    
    # Note: treatment_ai_content is defined based on tag containing 'ai content'
    # - treatment (1): posts with tag containing 'ai content'
    # - control (0): posts similar to AI content but NOT tagged as 'ai content'
    # This ensures we measure the true effect of the AI tag classification
    
    # 1. Data preprocessing
    print("\n=== 1. Data Preprocessing ===")
    
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
    
    # 2. Data analysis
    print("\n=== 2. Data Analysis ===")
    
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
    
    # Calculate uplift
    treatment_response_rate = df[df['treatment_ai_content'] == 1]['response'].mean()
    control_response_rate = df[df['treatment_ai_content'] == 0]['response'].mean()
    uplift = treatment_response_rate - control_response_rate
    
    print(f"\nUplift analysis:")
    print(f"  Treatment response rate: {treatment_response_rate:.2%}")
    print(f"  Control response rate: {control_response_rate:.2%}")
    print(f"  Uplift: {uplift:.2%}")
    
    # 3. Model training
    print("\n=== 3. Model Training ===")
    
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
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'model': model
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        
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
    
    # 4. Model comparison
    print("\n=== 4. Model Comparison ===")
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Precision': [results[model]['precision'] for model in results.keys()],
        'Recall': [results[model]['recall'] for model in results.keys()],
        'F1 Score': [results[model]['f1'] for model in results.keys()],
        'AUC': [results[model]['auc'] for model in results.keys()]
    })
    
    print(comparison_df.to_string(index=False))
    
    # 5. Best model analysis
    print("\n=== 5. Best Model Analysis ===")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    print(f"Best model: {best_model_name}")
    print(f"Best AUC: {results[best_model_name]['auc']:.4f}")
    
    # 6. Save results
    print("\n=== 6. Saving Results ===")
    
    # Save comparison results
    comparison_df.to_csv('output/model_comparison.csv', index=False)
    print("Model comparison saved to: output/model_comparison.csv")
    
    # Save best model
    import joblib
    joblib.dump(best_model, f'models/best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    print(f"Best model saved to: models/best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
    
    # Save feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': numeric_features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv('output/feature_importance.csv', index=False)
        print("Feature importance saved to: output/feature_importance.csv")
    
    # 7. Summary
    print("\n=== 7. Summary ===")
    
    print(f"Data volume: {len(df):,} samples")
    print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    print(f"Uplift: {uplift:.2%}")
    print(f"Best model: {best_model_name}")
    print(f"Best AUC: {results[best_model_name]['auc']:.4f}")
    
    return {
        'results': results,
        'best_model': best_model_name,
        'uplift': uplift,
        'comparison_df': comparison_df
    }

if __name__ == "__main__":
    results = main() 