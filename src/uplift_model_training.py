import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

class UpliftModelTraining:
    """Uplift model training and evaluation with optimized hyperparameter tuning"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_models = {}
        self.tuning_results = {}
        
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
            print(f"Missing required columns: {missing_cols}")
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
    
    def create_uplift_scorer(self, X, t, y):
        """Create custom scorer for uplift modeling"""
        def uplift_score(estimator, X_val, y_val):
            # Predict probabilities
            y_pred_proba = estimator.predict_proba(X_val)[:, 1]
            
            # Split by treatment
            treatment_mask = t.iloc[X_val.index] == 1
            control_mask = t.iloc[X_val.index] == 0
            
            if treatment_mask.sum() == 0 or control_mask.sum() == 0:
                return 0.0
            
            # Calculate uplift
            treatment_score = y_pred_proba[treatment_mask].mean()
            control_score = y_pred_proba[control_mask].mean()
            uplift = treatment_score - control_score
            
            return uplift
        
        return make_scorer(uplift_score, greater_is_better=True)
    
    def define_parameter_grids(self):
        """Define comprehensive parameter grids for different models"""
        
        # XGBoost parameter grid
        xgb_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Random Forest parameter grid
        rf_param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        return {
            'XGBoost': xgb_param_grid,
            'Random Forest': rf_param_grid
        }
    
    def define_parameter_distributions(self):
        """Define parameter distributions for randomized search"""
        
        # XGBoost parameter distributions
        xgb_param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 4, 5, 6],
            'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.05, 0.1, 0.5, 1.0]
        }
        
        # Random Forest parameter distributions
        rf_param_dist = {
            'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400],
            'max_depth': [5, 8, 10, 12, 15, 18, 20, None],
            'min_samples_split': [2, 3, 5, 8, 10, 15],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        return {
            'XGBoost': xgb_param_dist,
            'Random Forest': rf_param_dist
        }
    
    def optimized_hyperparameter_tuning(self, X_train, y_train, t_train, method='randomized', cv=5, n_iter=100, n_jobs=-1):
        """Perform optimized hyperparameter tuning"""
        print(f"\n=== Optimized Hyperparameter Tuning ({method.upper()}) ===")
        
        if method == 'grid':
            parameter_grids = self.define_parameter_grids()
            search_method = GridSearchCV
            search_params = {'param_grid': parameter_grids}
        elif method == 'randomized':
            parameter_distributions = self.define_parameter_distributions()
            search_method = RandomizedSearchCV
            search_params = {
                'param_distributions': parameter_distributions,
                'n_iter': n_iter,
                'random_state': 42
            }
        else:
            raise ValueError("Method must be 'grid' or 'randomized'")
        
        for model_name, param_config in search_params.items():
            print(f"\n--- Tuning {model_name} with {method.title()} Search ---")
            
            # Create model
            if model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, verbosity=0)
            else:
                base_model = RandomForestClassifier(random_state=42)
            
            # Create uplift scorer
            uplift_scorer = self.create_uplift_scorer(X_train, t_train, y_train)
            
            # Perform search
            search = search_method(
                estimator=base_model,
                scoring=uplift_scorer,
                cv=cv,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True,
                **{k: v for k, v in search_params.items() if k != 'param_distributions' and k != 'param_grid'}
            )
            
            if method == 'grid':
                search.param_grid = param_config
            else:
                search.param_distributions = param_config
            
            # Fit search
            search.fit(X_train, y_train)
            
            # Store results
            self.best_models[model_name] = search.best_estimator_
            self.tuning_results[model_name] = {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'cv_results': search.cv_results_
            }
            
            print(f"Best parameters: {search.best_params_}")
            print(f"Best uplift score: {search.best_score_:.4f}")
    
    def train_models(self, X_train, y_train, t_train, use_optimized_tuning=True, tuning_method='randomized'):
        """Train different uplift models with optional optimized tuning"""
        print("\n=== Model Training ===")
        
        if use_optimized_tuning:
            # Use optimized hyperparameter tuning
            self.optimized_hyperparameter_tuning(X_train, y_train, t_train, method=tuning_method, cv=5, n_iter=50)
            
            # Use best models from tuning
            self.models = self.best_models.copy()
        else:
            # Use default models (original approach)
            self.models = {
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, verbosity=0
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
            }
            
            # Train default models
            for model_name, model in self.models.items():
                print(f"\n=== Training {model_name} (Default Parameters) ===")
                model.fit(X_train.values, y_train.values)
                print(f"{model_name} trained successfully")
        
        self.results = {}
    
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
            
            # Calculate uplift score
            uplift_scorer = self.create_uplift_scorer(X_test, t_test, y_test)
            uplift_score = uplift_scorer(model, X_test, y_test)
            
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'uplift_score': uplift_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Uplift Score: {uplift_score:.4f}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    def analyze_feature_importance(self, numeric_features):
        """Analyze feature importance for best model"""
        print("\n=== Feature Importance Analysis ===")
        
        # Find best model based on uplift score
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['uplift_score'])
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
            'Uplift Score': [self.results[model]['uplift_score'] for model in self.results.keys()],
            'CV Accuracy': [self.results[model]['cv_mean'] for model in self.results.keys()],
            'CV Std': [self.results[model]['cv_std'] for model in self.results.keys()]
        })
        
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['Uplift Score'].idxmax()]
        print(f"\nBest model by uplift score: {best_model['Model']}")
        print(f"Uplift score: {best_model['Uplift Score']:.4f}")
        
        return comparison_df
    
    def save_results(self, output_file='uplift_model_eval.txt'):
        """Save model evaluation results"""
        print(f"\nSaving results to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("=== Uplift Model Evaluation Results ===\n\n")
            
            # Model comparison
            comparison_df = self.compare_models()
            f.write("Model Comparison:\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Best model details
            best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['uplift_score'])
            best_result = self.results[best_model_name]
            
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Uplift Score: {best_result['uplift_score']:.4f}\n")
            f.write(f"Accuracy: {best_result['accuracy']:.4f}\n")
            f.write(f"AUC: {best_result['auc']:.4f}\n")
            
            # Feature importance
            if hasattr(best_result['model'], 'feature_importances_'):
                f.write("\nTop 10 Feature Importance:\n")
                importance_df = self.analyze_feature_importance([col for col in self.models[best_model_name].feature_names_in_])
                if importance_df is not None:
                    for idx, row in importance_df.head(10).iterrows():
                        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
        
        print(f"Results saved to {output_file}")

def main():
    """Main execution function for optimized uplift model training"""
    print("=== Optimized Uplift Model Training ===\n")
    
    # Initialize trainer
    trainer = UpliftModelTraining()
    
    # Load data
    df = trainer.load_data('uplift_model_data.csv')
    if df is None:
        return
    
    # Preprocess data
    df, numeric_features, categorical_features = trainer.preprocess_data(df)
    if df is None:
        return
    
    # Analyze data
    analysis_results = trainer.analyze_data(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test, t_train, t_test = trainer.prepare_data(df, numeric_features)
    
    # Train models with optimized tuning
    trainer.train_models(X_train, y_train, t_train, use_optimized_tuning=True, tuning_method='randomized')
    
    # Evaluate models
    trainer.evaluate_models(X_test, y_test, t_test)
    
    # Analyze feature importance
    trainer.analyze_feature_importance(numeric_features)
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Save results
    trainer.save_results()
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main() 