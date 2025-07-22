import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class UpliftModelTraining:
    """Uplift model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_data(self, file_path):
        """Load uplift data"""
        print(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        print(f"Data shape: {df.shape}")
        return df
    
    def preprocess_data(self, df):
        """Preprocess data for uplift modeling"""
        print("Preprocessing data...")
        
        # Check for required columns
        required_cols = ['treatment_ai_content', 'response']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Note: treatment_ai_content is defined based on tag containing 'ai content'
        # - treatment (1): posts with tag containing 'ai content'
        # - control (0): posts similar to AI content but NOT tagged as 'ai content'
        # This ensures we measure the true effect of the AI tag classification
        
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
        
        return df, numeric_features, categorical_features
    
    def analyze_data(self, df):
        """Analyze data distributions and relationships"""
        print("\n=== Data Analysis ===")
        
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
        
        return {
            'treatment_dist': treatment_dist,
            'response_dist': response_dist,
            'uplift': uplift
        }
    
    def prepare_data(self, df, numeric_features):
        """Prepare data for model training"""
        print("\nPreparing data for training...")
        
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
        
        return X_train, X_test, y_train, y_test, t_train, t_test
    
    def train_models(self, X_train, y_train, t_train):
        """Train different uplift models"""
        print("\n=== Model Training ===")
        
        # Define models
        self.models = {
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        }
        
        self.results = {}
        
        for model_name, model in self.models.items():
            print(f"\n=== Training {model_name} ===")
            
            # Train model
            model.fit(X_train.values, y_train.values)
            
            # Store model
            self.models[model_name] = model
            
            print(f"✅ {model_name} trained successfully")
    
    def evaluate_models(self, X_test, y_test, t_test):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")
        
        for model_name, model in self.models.items():
            print(f"\n=== Evaluating {model_name} ===")
            
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
            cv_scores = cross_val_score(model, X_test.values, y_test.values, cv=5, scoring='accuracy')
            
            self.results[model_name] = {
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
    
    def analyze_feature_importance(self, numeric_features):
        """Analyze feature importance for best model"""
        print("\n=== Feature Importance Analysis ===")
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        best_model = self.results[best_model_name]['model']
        
        print(f"Analyzing feature importance for {best_model_name}")
        
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': numeric_features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 15 most important features:")
            for idx, row in importance_df.head(15).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        else:
            print("Model does not support feature importance")
            return None
    
    def compare_models(self):
        """Compare model performance"""
        print("\n=== Model Comparison ===")
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results.keys()],
            'Precision': [self.results[model]['precision'] for model in self.results.keys()],
            'Recall': [self.results[model]['recall'] for model in self.results.keys()],
            'F1 Score': [self.results[model]['f1'] for model in self.results.keys()],
            'AUC': [self.results[model]['auc'] for model in self.results.keys()],
            'CV Accuracy': [self.results[model]['cv_mean'] for model in self.results.keys()]
        })
        
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_results(self, comparison_df, importance_df=None):
        """Save results to files"""
        print("\n=== Saving Results ===")
        
        # Save comparison results
        comparison_df.to_csv('output/uplift_model_comparison.csv', index=False)
        print("Model comparison saved to: output/uplift_model_comparison.csv")
        
        # Save best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['auc'])
        best_model = self.results[best_model_name]['model']
        
        import joblib
        joblib.dump(best_model, f'models/best_uplift_model_{best_model_name.lower().replace(" ", "_")}.pkl')
        print(f"Best model saved to: models/best_uplift_model_{best_model_name.lower().replace(' ', '_')}.pkl")
        
        # Save feature importance
        if importance_df is not None:
            importance_df.to_csv('output/uplift_feature_importance.csv', index=False)
            print("Feature importance saved to: output/uplift_feature_importance.csv")
    
    def run_training_pipeline(self, file_path):
        """Complete training pipeline"""
        print("=== Uplift Model Training Pipeline ===\n")
        
        # Load data
        df = self.load_data(file_path)
        
        # Preprocess data
        preprocessed = self.preprocess_data(df)
        if preprocessed is None:
            return None
        
        df, numeric_features, categorical_features = preprocessed
        
        # Analyze data
        analysis_results = self.analyze_data(df)
        
        # Prepare data
        X_train, X_test, y_train, y_test, t_train, t_test = self.prepare_data(df, numeric_features)
        
        # Train models
        self.train_models(X_train, y_train, t_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test, t_test)
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance(numeric_features)
        
        # Compare models
        comparison_df = self.compare_models()
        
        # Save results
        self.save_results(comparison_df, importance_df)
        
        # Summary
        print("\n=== Training Pipeline Summary ===")
        print(f"Data volume: {len(df):,} samples")
        print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
        print(f"Uplift: {analysis_results['uplift']:.2%}")
        print(f"Best model: {max(self.results.keys(), key=lambda x: self.results[x]['auc'])}")
        print(f"Best AUC: {max(self.results.values(), key=lambda x: x['auc'])['auc']:.4f}")
        
        return {
            'results': self.results,
            'comparison_df': comparison_df,
            'importance_df': importance_df,
            'analysis_results': analysis_results
        }

def train_uplift_models(file_path):
    """Convenience function for training uplift models"""
    trainer = UpliftModelTraining()
    return trainer.run_training_pipeline(file_path)

if __name__ == "__main__":
    results = train_uplift_models('uplift_model_data.csv') 