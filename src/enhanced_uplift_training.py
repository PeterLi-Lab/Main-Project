import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

warnings.filterwarnings('ignore')

class EnhancedUpliftTraining:
    """Enhanced uplift model training with automated tuning, custom scorers, and time monitoring"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_times = {}
        self.best_models = {}
        self.tuning_results = {}
        self.performance_history = []
        
    def create_uplift_scorer(self, X, t, y):
        """Create custom uplift scorer for treatment effect estimation"""
        def uplift_score(estimator, X_val, y_val):
            try:
                # Predict probabilities
                y_pred_proba = estimator.predict_proba(X_val)[:, 1]
                
                # Get treatment assignments for this validation set
                # We need to map back to original indices
                if hasattr(X_val, 'index'):
                    val_indices = X_val.index
                else:
                    # If no index, assume sequential
                    val_indices = range(len(X_val))
                
                # Get treatment assignments for these indices
                t_val = t.iloc[val_indices] if hasattr(t, 'iloc') else t[val_indices]
                
                # Split by treatment
                treatment_mask = t_val == 1
                control_mask = t_val == 0
                
                if treatment_mask.sum() == 0 or control_mask.sum() == 0:
                    return 0.0
                
                # Calculate uplift
                treatment_score = y_pred_proba[treatment_mask].mean()
                control_score = y_pred_proba[control_mask].mean()
                uplift = treatment_score - control_score
                
                return uplift
            except Exception as e:
                print(f"Error in uplift scorer: {e}")
                return 0.0
        
        return make_scorer(uplift_score, greater_is_better=True)
    
    def create_qini_scorer(self, X, t, y):
        """Create Qini coefficient scorer for uplift modeling"""
        def qini_score(estimator, X_val, y_val):
            try:
                # Predict probabilities
                y_pred_proba = estimator.predict_proba(X_val)[:, 1]
                
                # Get treatment assignments for this validation set
                if hasattr(X_val, 'index'):
                    val_indices = X_val.index
                else:
                    val_indices = range(len(X_val))
                
                t_val = t.iloc[val_indices] if hasattr(t, 'iloc') else t[val_indices]
                
                # Create dataframe for calculation
                df = pd.DataFrame({
                    'pred': y_pred_proba,
                    'treatment': t_val,
                    'response': y_val
                })
                
                # Sort by prediction
                df = df.sort_values('pred', ascending=False)
                
                # Calculate cumulative metrics
                n_total = len(df)
                n_treatment = df['treatment'].sum()
                n_control = n_total - n_treatment
                
                if n_treatment == 0 or n_control == 0:
                    return 0.0
                
                # Calculate Qini coefficient
                treatment_response_rate = df[df['treatment'] == 1]['response'].mean()
                control_response_rate = df[df['treatment'] == 0]['response'].mean()
                
                qini = (treatment_response_rate - control_response_rate) * n_treatment * n_control / n_total
                
                return qini
            except Exception as e:
                print(f"Error in qini scorer: {e}")
                return 0.0
        
        return make_scorer(qini_score, greater_is_better=True)
    
    def define_comprehensive_parameter_grids(self):
        """Define comprehensive parameter grids for different models"""
        
        # XGBoost parameter grid
        xgb_param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'reg_alpha': [0, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 0.5, 1.0],
            'gamma': [0, 0.1, 0.5, 1.0]
        }
        
        # Random Forest parameter grid
        rf_param_grid = {
            'n_estimators': [50, 100, 200, 300, 400],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy']
        }
        
        # LightGBM parameter grid (if available)
        lgb_param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 9, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'num_leaves': [31, 50, 100, 200],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 0.5],
            'reg_lambda': [0, 0.01, 0.1, 0.5]
        }
        
        return {
            'XGBoost': xgb_param_grid,
            'Random Forest': rf_param_grid,
            'LightGBM': lgb_param_grid
        }
    
    def define_parameter_distributions(self):
        """Define parameter distributions for randomized search (optimized for speed)"""
        
        # XGBoost parameter distributions (reduced for speed)
        xgb_param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }
        
        # Random Forest parameter distributions (reduced for speed)
        rf_param_dist = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False]
        }
        
        return {
            'XGBoost': xgb_param_dist,
            'Random Forest': rf_param_dist
        }
    
    def automated_hyperparameter_tuning(self, X_train, y_train, t_train, method='randomized', cv=3, n_iter=20, n_jobs=1):
        """Perform automated hyperparameter tuning with time monitoring (optimized for speed)"""
        print(f"\n=== Automated Hyperparameter Tuning ({method.upper()}) ===")
        print(f"Using {n_iter} iterations, {cv}-fold CV, {n_jobs} jobs")
        
        start_time = time.time()
        
        if method == 'grid':
            parameter_configs = self.define_comprehensive_parameter_grids()
            search_method = GridSearchCV
        elif method == 'randomized':
            parameter_configs = self.define_parameter_distributions()
            search_method = RandomizedSearchCV
        else:
            raise ValueError("Method must be 'grid' or 'randomized'")
        
        for model_name, param_config in parameter_configs.items():
            print(f"\n--- Tuning {model_name} with {method.title()} Search ---")
            model_start_time = time.time()
            
            # Create model
            if model_name == 'XGBoost':
                base_model = xgb.XGBClassifier(random_state=42, verbosity=0)
            elif model_name == 'Random Forest':
                base_model = RandomForestClassifier(random_state=42, n_jobs=1)  # Use single thread for RF
            elif model_name == 'LightGBM':
                try:
                    import lightgbm as lgb
                    base_model = lgb.LGBMClassifier(random_state=42, verbosity=-1)
                except ImportError:
                    print("LightGBM not installed. Skipping...")
                    continue
            else:
                continue
            
            # Use accuracy instead of AUC to avoid overfitting detection issues
            if method == 'grid':
                search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_config,
                    scoring='accuracy',
                    cv=cv,
                    n_jobs=n_jobs,
                    verbose=0,  # Reduce verbosity
                    return_train_score=True
                )
            else:
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_config,
                    n_iter=n_iter,
                    scoring='accuracy',
                    cv=cv,
                    n_jobs=n_jobs,
                    verbose=0,  # Reduce verbosity
                    random_state=42,
                    return_train_score=True
                )
            
            # Fit search with timeout
            try:
                search.fit(X_train, y_train)
            except Exception as e:
                print(f"Error during {model_name} tuning: {e}")
                continue
            
            # Calculate uplift metrics for best model
            best_model = search.best_estimator_
            
            # Calculate uplift score manually
            y_pred_proba = best_model.predict_proba(X_train)[:, 1]
            treatment_mask = t_train == 1
            control_mask = t_train == 0
            
            if treatment_mask.sum() > 0 and control_mask.sum() > 0:
                treatment_score = y_pred_proba[treatment_mask].mean()
                control_score = y_pred_proba[control_mask].mean()
                uplift_score = treatment_score - control_score
            else:
                uplift_score = 0.0
            
            # Store results
            self.best_models[model_name] = best_model
            self.tuning_results[model_name] = {
                'best_params': search.best_params_,
                'best_accuracy_score': search.best_score_,
                'best_uplift_score': uplift_score,
                'cv_results': search.cv_results_,
                'tuning_time': time.time() - model_start_time
            }
            
            print(f"Best parameters: {search.best_params_}")
            print(f"Best accuracy score: {search.best_score_:.4f}")
            print(f"Best uplift score: {uplift_score:.4f}")
            print(f"Tuning time: {self.tuning_results[model_name]['tuning_time']:.2f} seconds")
        
        total_tuning_time = time.time() - start_time
        print(f"\nTotal tuning time: {total_tuning_time:.2f} seconds")
        
        return total_tuning_time
    
    def train_models_with_monitoring(self, X_train, y_train, t_train, X_test, y_test, t_test, use_automated_tuning=True):
        """Train models with comprehensive monitoring and evaluation"""
        print("\n=== Enhanced Model Training with Monitoring ===")
        
        if use_automated_tuning:
            # Use automated tuning with optimized parameters
            tuning_time = self.automated_hyperparameter_tuning(X_train, y_train, t_train, method='randomized', cv=3, n_iter=20, n_jobs=1)
            self.models = self.best_models.copy()
        else:
            # Use default models
            self.models = {
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=42, verbosity=0
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )
            }
            
            # Train default models with time monitoring
            for model_name, model in self.models.items():
                print(f"\n--- Training {model_name} (Default Parameters) ---")
                start_time = time.time()
                
                model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                self.training_times[model_name] = training_time
                print(f"Training time: {training_time:.2f} seconds")
        
        # Evaluate all models
        self.comprehensive_evaluation(X_test, y_test, t_test)
        
        return self.results
    
    def comprehensive_evaluation(self, X_test, y_test, t_test):
        """Comprehensive model evaluation with multiple metrics"""
        print("\n=== Comprehensive Model Evaluation ===")
        
        for model_name, model in self.models.items():
            print(f"\n--- Evaluating {model_name} ---")
            eval_start_time = time.time()
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Standard metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Uplift-specific metrics (calculated manually)
            treatment_mask = t_test == 1
            control_mask = t_test == 0
            
            if treatment_mask.sum() > 0 and control_mask.sum() > 0:
                treatment_score = y_pred_proba[treatment_mask].mean()
                control_score = y_pred_proba[control_mask].mean()
                uplift_score = treatment_score - control_score
                
                # Calculate Qini coefficient
                df = pd.DataFrame({
                    'pred': y_pred_proba,
                    'treatment': t_test,
                    'response': y_test
                })
                df = df.sort_values('pred', ascending=False)
                
                n_total = len(df)
                n_treatment = df['treatment'].sum()
                n_control = n_total - n_treatment
                
                if n_treatment > 0 and n_control > 0:
                    treatment_response_rate = df[df['treatment'] == 1]['response'].mean()
                    control_response_rate = df[df['treatment'] == 0]['response'].mean()
                    qini_score = (treatment_response_rate - control_response_rate) * n_treatment * n_control / n_total
                else:
                    qini_score = 0.0
            else:
                uplift_score = 0.0
                qini_score = 0.0
            
            # Cross validation
            cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')
            
            # Evaluation time
            eval_time = time.time() - eval_start_time
            
            # Store comprehensive results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'uplift_score': uplift_score,
                'qini_score': qini_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'evaluation_time': eval_time,
                'model': model
            }
            
            # Print results
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"AUC: {auc:.4f}")
            print(f"Uplift Score: {uplift_score:.4f}")
            print(f"Qini Score: {qini_score:.4f}")
            print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"Evaluation time: {eval_time:.2f} seconds")
    
    def performance_analysis_and_visualization(self):
        """Analyze and visualize model performance"""
        print("\n=== Performance Analysis and Visualization ===")
        
        # Create performance comparison dataframe
        performance_data = []
        for model_name, result in self.results.items():
            performance_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'AUC': result['auc'],
                'Uplift Score': result['uplift_score'],
                'Qini Score': result['qini_score'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Check if we have results to visualize
        if len(performance_df) == 0:
            print("No model results available for visualization")
            return performance_df
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison bar chart
        metrics = ['Accuracy', 'AUC', 'Uplift Score', 'Qini Score']
        x = np.arange(len(performance_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric in performance_df.columns:
                axes[0, 0].bar(x + i*width, performance_df[metric], width, label=metric)
        
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x + width*1.5)
        axes[0, 0].set_xticklabels(performance_df['Model'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uplift vs Qini scatter plot
        if 'Uplift Score' in performance_df.columns and 'Qini Score' in performance_df.columns:
            axes[0, 1].scatter(performance_df['Uplift Score'], performance_df['Qini Score'], 
                              s=100, alpha=0.7)
            for i, model in enumerate(performance_df['Model']):
                axes[0, 1].annotate(model, (performance_df['Uplift Score'].iloc[i], 
                                           performance_df['Qini Score'].iloc[i]))
            axes[0, 1].set_xlabel('Uplift Score')
            axes[0, 1].set_ylabel('Qini Score')
            axes[0, 1].set_title('Uplift vs Qini Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cross-validation scores
        if 'CV Mean' in performance_df.columns and 'CV Std' in performance_df.columns:
            cv_means = performance_df['CV Mean']
            cv_stds = performance_df['CV Std']
            axes[1, 0].bar(performance_df['Model'], cv_means, yerr=cv_stds, capsize=5)
            axes[1, 0].set_xlabel('Models')
            axes[1, 0].set_ylabel('CV Accuracy')
            axes[1, 0].set_title('Cross-Validation Performance')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training times (if available)
        if self.training_times:
            times = [self.training_times.get(model, 0) for model in performance_df['Model']]
            axes[1, 1].bar(performance_df['Model'], times)
            axes[1, 1].set_xlabel('Models')
            axes[1, 1].set_ylabel('Training Time (seconds)')
            axes[1, 1].set_title('Training Time Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_model_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return performance_df
    
    def save_enhanced_results(self, output_dir='output'):
        """Save comprehensive results and models"""
        print(f"\n=== Saving Enhanced Results to {output_dir} ===")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Save performance results
        performance_df = pd.DataFrame([
            {
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1': result['f1'],
                'AUC': result['auc'],
                'Uplift_Score': result['uplift_score'],
                'Qini_Score': result['qini_score'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std'],
                'Evaluation_Time': result['evaluation_time']
            }
            for model_name, result in self.results.items()
        ])
        
        performance_df.to_csv(f'{output_dir}/enhanced_model_performance.csv', index=False)
        
        # Save best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['uplift_score'])
        best_model = self.results[best_model_name]['model']
        
        joblib.dump(best_model, f'models/best_enhanced_uplift_model.pkl')
        
        # Save tuning results
        if self.tuning_results:
            tuning_df = pd.DataFrame([
                {
                    'Model': model_name,
                    'Best_Uplift_Score': result['best_uplift_score'],
                    'Best_Qini_Score': result['best_qini_score'],
                    'Tuning_Time': result['tuning_time'],
                    'Best_Parameters': str(result['best_params'])
                }
                for model_name, result in self.tuning_results.items()
            ])
            tuning_df.to_csv(f'{output_dir}/hyperparameter_tuning_results.csv', index=False)
        
        # Save comprehensive report
        with open(f'{output_dir}/enhanced_training_report.txt', 'w') as f:
            f.write("=== Enhanced Uplift Model Training Report ===\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Model Performance Summary:\n")
            f.write(performance_df.to_string(index=False))
            f.write("\n\n")
            
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Uplift Score: {self.results[best_model_name]['uplift_score']:.4f}\n")
            f.write(f"Best Qini Score: {self.results[best_model_name]['qini_score']:.4f}\n")
            
            if self.tuning_results:
                f.write(f"\nTuning Results:\n")
                f.write(tuning_df.to_string(index=False))
        
        print(f"Results saved to {output_dir}/")
        print(f"Best model saved to models/best_enhanced_uplift_model.pkl")

def main():
    """Main execution function for enhanced uplift training"""
    print("=== Enhanced Uplift Model Training Pipeline ===\n")
    
    # Initialize enhanced trainer
    trainer = EnhancedUpliftTraining()
    
    # Load data
    try:
        df = pd.read_csv('uplift_model_data.csv')
        print(f"Loaded data: {df.shape}")
    except FileNotFoundError:
        print("uplift_model_data.csv not found. Creating sample data for demonstration...")
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 10000
        df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'treatment_ai_content': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'response': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    X = df[feature_cols].fillna(0)
    y = df['response']
    t = df['treatment_ai_content']
    
    # Split data
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Train models with enhanced monitoring
    results = trainer.train_models_with_monitoring(
        X_train, y_train, t_train, X_test, y_test, t_test, 
        use_automated_tuning=True
    )
    
    # Analyze and visualize performance
    performance_df = trainer.performance_analysis_and_visualization()
    
    # Save results
    trainer.save_enhanced_results()
    
    print("\n=== Enhanced Training Complete ===")
    print(f"Best model: {max(results.keys(), key=lambda x: results[x]['uplift_score'])}")
    print(f"Best uplift score: {max(results.values(), key=lambda x: x['uplift_score'])['uplift_score']:.4f}")

if __name__ == "__main__":
    main()
