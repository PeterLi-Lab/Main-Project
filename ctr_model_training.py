#!/usr/bin/env python3
"""
CTR Model Training Script
Trains CTR prediction models using user-post click samples
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

class CTRModelTrainer:
    def __init__(self):
        """Initialize CTR model trainer"""
        self.models = {}
        self.results = {}
        
    def load_ctr_data(self, ctr_samples_path='user_post_click_samples.csv'):
        """Load CTR samples and merge with features"""
        print("=== Loading CTR Data ===")
        
        # Load CTR samples
        self.df_ctr = pd.read_csv(ctr_samples_path)
        print(f"Loaded {len(self.df_ctr)} CTR samples")
        print(f"Columns: {self.df_ctr.columns.tolist()}")
        
        # Sample data if too large (for faster training)
        if len(self.df_ctr) > 50000:
            print(f"Sampling 50,000 samples for faster training...")
            self.df_ctr = self.df_ctr.sample(n=50000, random_state=42)
            print(f"Sampled to {len(self.df_ctr)} samples")
        
        # Load user and post features (from data_preprocessing output)
        try:
            # Try to load from data_preprocessing output
            from data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            df_combined, _, _, _ = preprocessor.run_full_pipeline()
            
            # Sample df_combined if too large
            if len(df_combined) > 100000:
                df_combined = df_combined.sample(n=100000, random_state=42)
            
            # Merge with CTR samples
            print("Merging with post features...")
            self.df_ctr = self.df_ctr.merge(
                df_combined[['Id', 'Title', 'Body', 'Tags', 'CreationDate', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount']],
                left_on='post_id', right_on='Id', how='left'
            )
            
            # Add user features
            print("Merging with user features...")
            user_features = df_combined[['OwnerUserId', 'Reputation', 'user_post_count', 'user_account_age_days']].drop_duplicates()
            self.df_ctr = self.df_ctr.merge(
                user_features,
                left_on='user_id', right_on='OwnerUserId', how='left'
            )
            
        except Exception as e:
            print(f"Warning: Could not load full features, using basic features: {e}")
            # Use basic features from CTR samples
            pass
        
        print(f"Final CTR dataset shape: {self.df_ctr.shape}")
        return True
    
    def create_features(self):
        """Create features for CTR prediction"""
        print("\n=== Creating Features ===")
        
        # 定义特征处理任务
        feature_tasks = [
            ('post_title_length', 'post_title', lambda x: x.str.len().fillna(0)),
            ('post_body_length', 'post_body', lambda x: x.str.len().fillna(0)),
            ('post_age_days', 'post_creation_date', lambda x: (pd.Timestamp.now() - pd.to_datetime(x)).dt.days if x.notnull().all() else 0),
            ('user_reputation', 'user_reputation', lambda x: x.fillna(0)),
            ('user_post_count', 'user_post_count', lambda x: x.fillna(0)),
            ('user_account_age_days', 'user_account_age_days', lambda x: x.fillna(0)),
            ('post_score', 'post_score', lambda x: x.fillna(0)),
            ('post_view_count', 'post_view_count', lambda x: x.fillna(0)),
            ('post_answer_count', 'post_answer_count', lambda x: x.fillna(0)),
            ('post_comment_count', 'post_comment_count', lambda x: x.fillna(0)),
        ]
        
        for feat_name, col_name, func in tqdm(feature_tasks, desc="Feature engineering"):
            if col_name in self.df_ctr.columns:
                if feat_name == 'post_age_days':
                    # 日期健壮处理
                    mask = (self.df_ctr[col_name] != "0") & (self.df_ctr[col_name].notnull())
                    self.df_ctr[feat_name] = 0
                    self.df_ctr.loc[mask, feat_name] = (
                        pd.Timestamp.now() - pd.to_datetime(self.df_ctr.loc[mask, col_name], errors='coerce')
                    ).dt.days
                else:
                    self.df_ctr[feat_name] = func(self.df_ctr[col_name])
            else:
                self.df_ctr[feat_name] = 0
        
        # interest_score 特征
        if 'interest_score' not in self.df_ctr.columns:
            self.df_ctr['interest_score'] = 0
        
        self.feature_columns = [
            'post_title_length', 'post_body_length', 'post_age_days',
            'user_reputation', 'user_post_count', 'user_account_age_days',
            'post_score', 'post_view_count', 'post_answer_count', 'post_comment_count',
            'interest_score'
        ]
        
        # 填充所有特征的缺失值
        for col in self.feature_columns:
            if col in self.df_ctr.columns:
                self.df_ctr[col] = self.df_ctr[col].fillna(0)
            else:
                self.df_ctr[col] = 0
        print(f"Created {len(self.feature_columns)} features")
        return True
    
    def train_models(self):
        """Train multiple CTR prediction models"""
        print("\n=== Training CTR Models ===")
        
        # Prepare features and target
        X = self.df_ctr[self.feature_columns].fillna(0)
        y = self.df_ctr['is_click']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Positive class ratio: {y.mean():.3f}")
        
        # Define models (simplified for faster training)
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=500),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=50)
        }
        
        # Train and evaluate each model
        for name, model in tqdm(models.items(), desc="Training models"):
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  AUC: {auc:.4f}")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
        
        return True
    
    def print_results(self):
        """Print model comparison results"""
        print("\n=== CTR Model Results ===")
        
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': results['accuracy'],
                'AUC': results['auc']
            }
            for name, results in self.results.items()
        ])
        
        print(results_df.to_string(index=False))
        
        # Best model
        if not results_df.empty:
            best_model = results_df.loc[results_df['AUC'].idxmax()]
            print(f"\nBest model: {best_model['Model']} (AUC: {best_model['AUC']:.4f})")
        
        return True
    
    def save_models(self, output_dir='models'):
        """Save trained models"""
        print(f"\n=== Saving Models ===")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = f"{output_dir}/ctr_{name}.pkl"
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {model_path}")
        
        # Save feature columns
        feature_path = f"{output_dir}/ctr_feature_columns.pkl"
        with open(feature_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        print(f"Saved feature columns to {feature_path}")
        
        return True
    
    def run_full_pipeline(self):
        """Run complete CTR model training pipeline"""
        print("=== CTR Model Training Pipeline ===")
        
        # Load data
        self.load_ctr_data()
        
        # Create features
        self.create_features()
        
        # Train models
        self.train_models()
        
        # Print results
        self.print_results()
        
        # Save models
        self.save_models()
        
        print("\n=== CTR Training Complete ===")
        return self.models, self.results

def main():
    """Main function"""
    trainer = CTRModelTrainer()
    models, results = trainer.run_full_pipeline()
    return models, results

if __name__ == "__main__":
    main() 