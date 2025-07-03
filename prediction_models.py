#!/usr/bin/env python3
"""
Prediction Models for Stack Overflow Data Analysis
Includes CTR prediction and user retention prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class CTRPredictor:
    """Click-Through Rate Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None
        
    def prepare_ctr_features(self, df_combined):
        """Prepare features for CTR prediction"""
        print("=== Preparing CTR Prediction Features ===")
        
        # 1. Basic post features
        ctr_features = []
        
        # Text features
        if 'title_length' in df_combined.columns:
            ctr_features.append('title_length')
        if 'post_length' in df_combined.columns:
            ctr_features.append('post_length')
        if 'num_tags' in df_combined.columns:
            ctr_features.append('num_tags')
        
        # Time features
        if 'post_age_days' in df_combined.columns:
            ctr_features.append('post_age_days')
        
        # User features
        if 'user_post_count' in df_combined.columns:
            ctr_features.append('user_post_count')
        if 'user_reputation' in df_combined.columns:
            ctr_features.append('user_reputation')
        
        # 2. Influence features
        influence_features = [
            'total_influence_score', 'high_quality_influence', 'influence_domains_count',
            'gold_badges', 'silver_badges', 'bronze_badges', 'vote_ratio'
        ]
        
        for feature in influence_features:
            if feature in df_combined.columns:
                ctr_features.append(feature)
        
        # 3. Badge features
        badge_features = [
            'total_badges', 'unique_badge_types', 'badge_quality_score',
            'badge_rate_per_day', 'recent_badges_30d', 'is_badge_active'
        ]
        
        for feature in badge_features:
            if feature in df_combined.columns:
                ctr_features.append(feature)
        
        # 4. Content quality features
        quality_features = [
            'Score', 'ViewCount', 'AnswerCount', 'CommentCount'
        ]
        
        for feature in quality_features:
            if feature in df_combined.columns:
                ctr_features.append(feature)
        
        # 5. Tag features
        tag_features = [col for col in df_combined.columns if col.startswith('has_badge_')]
        ctr_features.extend(tag_features)
        
        # 6. Domain features
        domain_features = [col for col in df_combined.columns if col.startswith('influence_') and not col.endswith('_scaled')]
        ctr_features.extend(domain_features)
        
        # 7. Categorical feature encoding
        categorical_features = [
            'influence_level', 'multi_domain_influence', 'badge_level', 'badge_quality_level'
        ]
        
        label_encoders = {}
        for feature in categorical_features:
            if feature in df_combined.columns:
                le = LabelEncoder()
                df_combined[f'{feature}_encoded'] = le.fit_transform(df_combined[feature].fillna('Unknown'))
                ctr_features.append(f'{feature}_encoded')
                label_encoders[feature] = le
        
        # 8. Interaction features
        if 'total_influence_score' in df_combined.columns and 'user_post_count' in df_combined.columns:
            df_combined['influence_post_ratio'] = df_combined['total_influence_score'] / (df_combined['user_post_count'] + 1)
            ctr_features.append('influence_post_ratio')
        
        if 'total_influence_score' in df_combined.columns and 'post_age_days' in df_combined.columns:
            df_combined['influence_age_ratio'] = df_combined['total_influence_score'] / (df_combined['post_age_days'] + 1)
            ctr_features.append('influence_age_ratio')
        
        # 9. Composite features
        if 'vote_ratio' in df_combined.columns and 'total_influence_score' in df_combined.columns:
            df_combined['quality_influence_score'] = df_combined['vote_ratio'] * df_combined['total_influence_score']
            ctr_features.append('quality_influence_score')
        
        print(f"Prepared {len(ctr_features)} features for CTR prediction")
        print(f"Feature list: {ctr_features[:10]}...")  # Show first 10 features
        
        return df_combined, ctr_features, label_encoders
    
    def train_ctr_model(self, df_combined, target_col='ctr_proxy_normalized', model_type='xgboost'):
        """Train CTR prediction model"""
        print(f"\n=== Training CTR Prediction Model ({model_type}) ===")
        
        # Prepare features
        df_combined, ctr_features, label_encoders = self.prepare_ctr_features(df_combined)
        
        # Check target variable
        if target_col not in df_combined.columns:
            print(f"Target column '{target_col}' not found. Available columns: {df_combined.columns.tolist()}")
            return None
        
        # Prepare data
        X = df_combined[ctr_features].fillna(0)
        y = df_combined[target_col].fillna(0)
        
        # Remove infinite and NaN values
        X = X.replace([np.inf, -np.inf], 0)
        y = y.replace([np.inf, -np.inf], 0)
        
        # Feature selection
        if len(ctr_features) > 50:
            print("Performing feature selection...")
            selector = SelectKBest(score_func=f_regression, k=50)
            X_selected = selector.fit_transform(X, y)
            selected_features = [ctr_features[i] for i in selector.get_support(indices=True)]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            ctr_features = selected_features
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select model
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': ctr_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        
        self.feature_names = ctr_features
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'mse': mse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_ctr(self, df_new):
        """Predict CTR for new data"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # 准备特征
        df_new, ctr_features, _ = self.prepare_ctr_features(df_new)
        
        # 确保特征顺序一致
        missing_features = set(self.feature_names) - set(ctr_features)
        for feature in missing_features:
            df_new[feature] = 0
        
        X_new = df_new[self.feature_names].fillna(0)
        X_new = X_new.replace([np.inf, -np.inf], 0)
        
        # 标准化
        X_new_scaled = self.scaler.transform(X_new)
        
        # 预测
        predictions = self.model.predict(X_new_scaled)
        
        return predictions
    
    def visualize_ctr_results(self, results):
        """Visualize CTR prediction results"""
        if results is None:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. 预测 vs 实际值
        plt.subplot(2, 3, 1)
        plt.scatter(results['y_test'], results['y_pred'], alpha=0.5)
        plt.plot([results['y_test'].min(), results['y_test'].max()], 
                [results['y_test'].min(), results['y_test'].max()], 'r--', lw=2)
        plt.xlabel('Actual CTR')
        plt.ylabel('Predicted CTR')
        plt.title(f'CTR Prediction (R² = {results["r2"]:.3f})')
        
        # 2. 残差图
        plt.subplot(2, 3, 2)
        residuals = results['y_test'] - results['y_pred']
        plt.scatter(results['y_pred'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted CTR')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # 3. 特征重要性
        if results['feature_importance'] is not None:
            plt.subplot(2, 3, 3)
            top_features = results['feature_importance'].head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance')
        
        # 4. 预测分布
        plt.subplot(2, 3, 4)
        plt.hist(results['y_pred'], bins=30, alpha=0.7, label='Predicted')
        plt.hist(results['y_test'], bins=30, alpha=0.7, label='Actual')
        plt.xlabel('CTR')
        plt.ylabel('Frequency')
        plt.title('CTR Distribution')
        plt.legend()
        
        # 5. 误差分布
        plt.subplot(2, 3, 5)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        
        # 6. 模型性能指标
        plt.subplot(2, 3, 6)
        metrics = ['MSE', 'R²']
        values = [results['mse'], results['r2']]
        plt.bar(metrics, values)
        plt.ylabel('Value')
        plt.title('Model Performance Metrics')
        
        plt.tight_layout()
        plt.show()

class RetentionPredictor:
    """User Retention Prediction Model"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.feature_names = None
        
    def prepare_retention_features(self, df_combined, retention_window_days=30):
        """Prepare features for retention prediction"""
        print("=== Preparing Retention Prediction Features ===")
        
        # 1. 用户活跃度特征
        retention_features = []
        
        # 基础活跃度
        if 'user_post_count' in df_combined.columns:
            retention_features.append('user_post_count')
        
        if 'post_age_days' in df_combined.columns:
            retention_features.append('post_age_days')
        
        # 2. 影响力特征
        influence_features = [
            'total_influence_score', 'high_quality_influence', 'influence_domains_count',
            'gold_badges', 'silver_badges', 'bronze_badges', 'vote_ratio'
        ]
        
        for feature in influence_features:
            if feature in df_combined.columns:
                retention_features.append(feature)
        
        # 3. 徽章特征
        badge_features = [
            'total_badges', 'unique_badge_types', 'badge_quality_score',
            'badge_rate_per_day', 'recent_badges_30d', 'is_badge_active'
        ]
        
        for feature in badge_features:
            if feature in df_combined.columns:
                retention_features.append(feature)
        
        # 4. 内容质量特征
        quality_features = [
            'Score', 'ViewCount', 'AnswerCount', 'CommentCount'
        ]
        
        for feature in quality_features:
            if feature in df_combined.columns:
                retention_features.append(feature)
        
        # 5. 用户行为特征
        if 'first_badge_date' in df_combined.columns:
            df_combined['days_since_first_badge'] = (
                pd.Timestamp.now() - df_combined['first_badge_date']
            ).dt.days
            retention_features.append('days_since_first_badge')
        
        # 6. 参与度特征
        if 'AnswerCount' in df_combined.columns and 'CommentCount' in df_combined.columns:
            df_combined['engagement_score'] = df_combined['AnswerCount'] + df_combined['CommentCount']
            retention_features.append('engagement_score')
        
        # 7. 分类特征编码
        categorical_features = [
            'influence_level', 'multi_domain_influence', 'badge_level', 'badge_quality_level'
        ]
        
        label_encoders = {}
        for feature in categorical_features:
            if feature in df_combined.columns:
                le = LabelEncoder()
                df_combined[f'{feature}_encoded'] = le.fit_transform(df_combined[feature].fillna('Unknown'))
                retention_features.append(f'{feature}_encoded')
                label_encoders[feature] = le
        
        # 8. 创建留存目标变量（示例）
        # 这里需要根据实际数据创建留存标签
        # 假设：如果用户在过去30天内有活动，则认为是留存用户
        if 'recent_badges_30d' in df_combined.columns:
            df_combined['is_retained'] = (df_combined['recent_badges_30d'] > 0).astype(int)
        else:
            # 如果没有recent_badges_30d，使用其他指标
            if 'is_badge_active' in df_combined.columns:
                df_combined['is_retained'] = df_combined['is_badge_active'].astype(int)
            else:
                # 默认假设所有用户都是留存用户（用于演示）
                df_combined['is_retained'] = 1
        
        print(f"Prepared {len(retention_features)} features for retention prediction")
        print(f"Retention rate: {df_combined['is_retained'].mean():.3f}")
        
        return df_combined, retention_features, label_encoders
    
    def train_retention_model(self, df_combined, target_col='is_retained', model_type='xgboost'):
        """Train retention prediction model"""
        print(f"\n=== Training Retention Prediction Model ({model_type}) ===")
        
        # 准备特征
        df_combined, retention_features, label_encoders = self.prepare_retention_features(df_combined)
        
        # 准备数据
        X = df_combined[retention_features].fillna(0)
        y = df_combined[target_col]
        
        # 移除无穷大和NaN值
        X = X.replace([np.inf, -np.inf], 0)
        
        # 特征选择
        if len(retention_features) > 30:
            print("Performing feature selection...")
            selector = SelectKBest(score_func=f_classif, k=30)
            X_selected = selector.fit_transform(X, y)
            selected_features = [retention_features[i] for i in selector.get_support(indices=True)]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            retention_features = selected_features
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 选择模型
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 训练模型
        self.model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': retention_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(self.feature_importance.head(10))
        
        self.feature_names = retention_features
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'accuracy': accuracy,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def predict_retention(self, df_new):
        """Predict retention for new data"""
        if self.model is None:
            print("Model not trained yet. Please train the model first.")
            return None
        
        # 准备特征
        df_new, retention_features, _ = self.prepare_retention_features(df_new)
        
        # 确保特征顺序一致
        missing_features = set(self.feature_names) - set(retention_features)
        for feature in missing_features:
            df_new[feature] = 0
        
        X_new = df_new[self.feature_names].fillna(0)
        X_new = X_new.replace([np.inf, -np.inf], 0)
        
        # 标准化
        X_new_scaled = self.scaler.transform(X_new)
        
        # 预测
        predictions = self.model.predict(X_new_scaled)
        predictions_proba = self.model.predict_proba(X_new_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        return predictions, predictions_proba
    
    def visualize_retention_results(self, results):
        """Visualize retention prediction results"""
        if results is None:
            return
        
        plt.figure(figsize=(15, 10))
        
        # 1. 混淆矩阵
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 2. 特征重要性
        if results['feature_importance'] is not None:
            plt.subplot(2, 3, 2)
            top_features = results['feature_importance'].head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Feature Importance')
        
        # 3. 预测概率分布
        if results['y_pred_proba'] is not None:
            plt.subplot(2, 3, 3)
            plt.hist(results['y_pred_proba'][results['y_test'] == 0], bins=20, alpha=0.7, label='Not Retained')
            plt.hist(results['y_pred_proba'][results['y_test'] == 1], bins=20, alpha=0.7, label='Retained')
            plt.xlabel('Retention Probability')
            plt.ylabel('Frequency')
            plt.title('Prediction Probability Distribution')
            plt.legend()
        
        # 4. 模型性能指标
        plt.subplot(2, 3, 4)
        metrics = ['Accuracy']
        values = [results['accuracy']]
        plt.bar(metrics, values)
        plt.ylabel('Value')
        plt.title('Model Performance Metrics')
        
        # 5. 留存率分析
        plt.subplot(2, 3, 5)
        retention_rates = [
            results['y_test'].mean(),
            results['y_pred'].mean()
        ]
        labels = ['Actual', 'Predicted']
        plt.bar(labels, retention_rates)
        plt.ylabel('Retention Rate')
        plt.title('Retention Rate Comparison')
        
        # 6. 预测vs实际
        plt.subplot(2, 3, 6)
        plt.scatter(range(len(results['y_test'])), results['y_test'], alpha=0.5, label='Actual')
        plt.scatter(range(len(results['y_pred'])), results['y_pred'], alpha=0.5, label='Predicted')
        plt.xlabel('Sample Index')
        plt.ylabel('Retention Status')
        plt.title('Prediction vs Actual')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

class IndustrialCTRPredictor:
    """Industrial-Grade CTR Prediction Models
    Implements models used by major internet companies (Alibaba, ByteDance, Google, Meta)
    """
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.feature_encoder = None
        self.online_metrics = {}
        
    def prepare_industrial_features(self, df_combined):
        """Prepare features for industrial CTR models"""
        print("=== Preparing Industrial CTR Features ===")
        
        # Get all numerical and categorical features
        numerical_features = []
        categorical_features = []
        
        # Numerical features
        numerical_cols = ['Score', 'ViewCount', 'AnswerCount', 'CommentCount', 
                         'title_length', 'post_length', 'num_tags', 'post_age_days',
                         'user_post_count', 'user_reputation', 'total_votes', 'upvotes',
                         'vote_ratio', 'total_influence_score', 'high_quality_influence',
                         'total_badges', 'badge_quality_score', 'badge_rate_per_day']
        
        for col in numerical_cols:
            if col in df_combined.columns:
                numerical_features.append(col)
        
        # Categorical features (hash encoded)
        categorical_cols = [col for col in df_combined.columns if col.endswith('_hash')]
        categorical_features.extend(categorical_cols)
        
        # Cross features
        cross_cols = [col for col in df_combined.columns if col.endswith('_cross')]
        categorical_features.extend(cross_cols)
        
        # Sequence features
        seq_cols = [col for col in df_combined.columns if col.startswith('seq_')]
        numerical_features.extend(seq_cols)
        
        # Context features
        context_cols = ['hour_of_day', 'day_of_week', 'month_of_year', 'is_weekend', 'is_peak_hours']
        for col in context_cols:
            if col in df_combined.columns:
                numerical_features.append(col)
        
        # Interaction features
        interaction_cols = ['quality_view_ratio', 'vote_quality_ratio', 'engagement_efficiency', 
                           'content_complexity', 'score_per_day']
        for col in interaction_cols:
            if col in df_combined.columns:
                numerical_features.append(col)
        
        # Normalize numerical features
        if numerical_features:
            scaler = StandardScaler()
            df_combined[numerical_features] = scaler.fit_transform(df_combined[numerical_features].fillna(0))
            self.feature_encoder = scaler
        
        print(f"Prepared {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
        
        return df_combined, numerical_features, categorical_features
    
    def train_industrial_models(self, df_combined, target_col='ctr_proxy_normalized'):
        """Train multiple industrial CTR models"""
        print("\n=== Training Industrial CTR Models ===")
        
        # Prepare features
        df_combined, numerical_features, categorical_features = self.prepare_industrial_features(df_combined)
        
        # Prepare target
        y = (df_combined[target_col] >= df_combined[target_col].quantile(0.7)).astype(int)
        
        # Combine all features
        all_features = numerical_features + categorical_features
        X = df_combined[all_features].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train models
        models_to_train = ['lr', 'fm', 'deepfm', 'dcn', 'din']
        
        for model_name in models_to_train:
            try:
                print(f"Training {model_name.upper()} model...")
                model = self._train_single_industrial_model(model_name, X_train, y_train, X_test, y_test)
                if model is not None:
                    self.models[model_name] = model
                    
                    # Evaluate model
                    y_pred_proba = self._predict_industrial_model(model_name, X_test)
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    logloss_score = log_loss(y_test, y_pred_proba)
                    
                    self.model_performance[model_name] = {
                        'auc': auc_score,
                        'logloss': logloss_score,
                        'feature_count': X_train.shape[1]
                    }
                    
                    print(f"{model_name.upper()} - AUC: {auc_score:.4f}, LogLoss: {logloss_score:.4f}")
                    
            except Exception as e:
                print(f"Failed to train {model_name}: {e}")
        
        # Save best model
        self._save_best_industrial_model()
        
        return self.models
    
    def _train_single_industrial_model(self, model_name, X_train, y_train, X_test, y_test):
        """Train single industrial model"""
        if model_name == 'lr':
            return self._train_lr_model(X_train, y_train)
        elif model_name == 'fm':
            return self._train_fm_model(X_train, y_train)
        elif model_name == 'deepfm':
            return self._train_deepfm_model(X_train, y_train, X_test, y_test)
        elif model_name == 'dcn':
            return self._train_dcn_model(X_train, y_train, X_test, y_test)
        elif model_name == 'din':
            return self._train_din_model(X_train, y_train, X_test, y_test)
        else:
            print(f"Unknown model: {model_name}")
            return None
    
    def _train_lr_model(self, X_train, y_train):
        """Train Logistic Regression (baseline)"""
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def _train_fm_model(self, X_train, y_train):
        """Train Factorization Machine (simplified)"""
        # Use Random Forest as simplified FM
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def _train_deepfm_model(self, X_train, y_train, X_test, y_test):
        """Train DeepFM model"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
            
            input_dim = X_train.shape[1]
            
            # Input layer
            inputs = layers.Input(shape=(input_dim,))
            
            # FM part (linear + pairwise interactions)
            fm_linear = layers.Dense(1, activation='linear')(inputs)
            
            # Deep part
            deep = layers.Dense(128, activation='relu')(inputs)
            deep = layers.Dropout(0.3)(deep)
            deep = layers.Dense(64, activation='relu')(deep)
            deep = layers.Dropout(0.3)(deep)
            deep = layers.Dense(32, activation='relu')(deep)
            
            # Combine FM and Deep
            combined = layers.Concatenate()([fm_linear, deep])
            outputs = layers.Dense(1, activation='sigmoid')(combined)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Training
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=1024,
                epochs=50,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model
            
        except ImportError:
            print("TensorFlow not available, skipping DeepFM")
            return None
        except Exception as e:
            print(f"DeepFM training failed: {e}")
            return None
    
    def _train_dcn_model(self, X_train, y_train, X_test, y_test):
        """Train Deep & Cross Network"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
            
            input_dim = X_train.shape[1]
            
            # Input layer
            inputs = layers.Input(shape=(input_dim,))
            
            # Cross Network
            cross_layer = inputs
            for i in range(3):  # 3 cross layers
                cross_layer = layers.Dense(input_dim, activation='linear')(cross_layer)
                cross_layer = layers.Add()([cross_layer, inputs])
            
            # Deep Network
            deep_layer = layers.Dense(128, activation='relu')(inputs)
            deep_layer = layers.Dense(64, activation='relu')(deep_layer)
            
            # Combine
            combined = layers.Concatenate()([cross_layer, deep_layer])
            outputs = layers.Dense(1, activation='sigmoid')(combined)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=1024,
                epochs=50,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model
            
        except ImportError:
            print("TensorFlow not available, skipping DCN")
            return None
        except Exception as e:
            print(f"DCN training failed: {e}")
            return None
    
    def _train_din_model(self, X_train, y_train, X_test, y_test):
        """Train Deep Interest Network"""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
            
            input_dim = X_train.shape[1]
            
            # Input layer
            inputs = layers.Input(shape=(input_dim,))
            
            # Attention mechanism (simplified)
            attention_weights = layers.Dense(64, activation='relu')(inputs)
            attention_weights = layers.Dense(32, activation='relu')(attention_weights)
            attention_weights = layers.Dense(1, activation='sigmoid')(attention_weights)
            
            # Weighted features
            weighted_features = layers.Multiply()([inputs, attention_weights])
            
            # Deep Network
            deep_layer = layers.Dense(128, activation='relu')(weighted_features)
            deep_layer = layers.Dropout(0.3)(deep_layer)
            deep_layer = layers.Dense(64, activation='relu')(deep_layer)
            deep_layer = layers.Dropout(0.3)(deep_layer)
            deep_layer = layers.Dense(32, activation='relu')(deep_layer)
            
            outputs = layers.Dense(1, activation='sigmoid')(deep_layer)
            
            model = models.Model(inputs=inputs, outputs=outputs)
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                batch_size=1024,
                epochs=50,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return model
            
        except ImportError:
            print("TensorFlow not available, skipping DIN")
            return None
        except Exception as e:
            print(f"DIN training failed: {e}")
            return None
    
    def _predict_industrial_model(self, model_name, X):
        """Predict using industrial model"""
        model = self.models.get(model_name)
        if model is None:
            return None
        
        if model_name in ['lr', 'fm']:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X).flatten()
    
    def _save_best_industrial_model(self):
        """Save best performing model"""
        if not self.model_performance:
            return
        
        # Find best model by AUC
        best_model = max(self.model_performance.items(), key=lambda x: x[1]['auc'])
        best_model_name = best_model[0]
        
        print(f"\nBest model: {best_model_name.upper()} (AUC: {best_model[1]['auc']:.4f})")
        
        # Save model
        import joblib
        import os
        
        os.makedirs("models", exist_ok=True)
        model_path = f"models/best_industrial_ctr_{best_model_name}.pkl"
        
        if best_model_name in ['lr', 'fm']:
            joblib.dump(self.models[best_model_name], model_path)
        else:
            self.models[best_model_name].save(f"models/best_industrial_ctr_{best_model_name}.h5")
        
        print(f"Best model saved to: {model_path}")
    
    def online_predict(self, features_dict, model_name=None):
        """Online prediction service"""
        import time
        
        start_time = time.time()
        
        try:
            # Select model
            if model_name is None:
                if not self.model_performance:
                    raise ValueError("No available models")
                best_model = max(self.model_performance.items(), key=lambda x: x[1]['auc'])
                model_name = best_model[0]
            
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Model {model_name} not found")
            
            # Prepare features
            feature_list = []
            for key, value in features_dict.items():
                feature_list.append(value)
            
            X = np.array(feature_list).reshape(1, -1)
            
            # Predict
            if model_name in ['lr', 'fm']:
                ctr_prob = model.predict_proba(X)[0, 1]
            else:
                ctr_prob = model.predict(X)[0]
            
            response_time = time.time() - start_time
            
            return {
                'ctr_probability': float(ctr_prob),
                'model_name': model_name,
                'response_time': response_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'response_time': time.time() - start_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def evaluate_industrial_models(self):
        """Evaluate all industrial models"""
        if not self.model_performance:
            print("No models to evaluate")
            return
        
        print("\n=== Industrial Model Performance Comparison ===")
        print(f"{'Model':<15} {'AUC':<10} {'LogLoss':<10} {'Features':<10}")
        print("-" * 50)
        
        for name, metrics in self.model_performance.items():
            print(f"{name.upper():<15} {metrics['auc']:<10.4f} {metrics['logloss']:<10.4f} {metrics['feature_count']:<10}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        model_names = list(self.model_performance.keys())
        auc_scores = [self.model_performance[name]['auc'] for name in model_names]
        
        ax1.bar(model_names, auc_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        ax1.set_title('Industrial Model AUC Comparison')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(auc_scores):
            ax1.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        # LogLoss comparison
        logloss_scores = [self.model_performance[name]['logloss'] for name in model_names]
        
        ax2.bar(model_names, logloss_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
        ax2.set_title('Industrial Model LogLoss Comparison')
        ax2.set_ylabel('LogLoss Score')
        for i, v in enumerate(logloss_scores):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('output/industrial_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def run_prediction_analysis(df_combined):
    """Run complete prediction analysis"""
    print("=== Running Complete Prediction Analysis ===")
    
    # 1. CTR Prediction
    print("\n1. CTR Prediction Analysis")
    ctr_predictor = CTRPredictor()
    
    # 尝试不同的模型
    ctr_models = ['xgboost', 'lightgbm', 'random_forest']
    ctr_results = {}
    
    for model_type in ctr_models:
        try:
            result = ctr_predictor.train_ctr_model(df_combined, model_type=model_type)
            if result:
                ctr_results[model_type] = result
                print(f"  {model_type.upper()}: R² = {result['r2']:.3f}, MSE = {result['mse']:.4f}")
        except Exception as e:
            print(f"  {model_type.upper()}: Error - {e}")
    
    # 选择最佳CTR模型
    if ctr_results:
        best_ctr_model = max(ctr_results.keys(), key=lambda x: ctr_results[x]['r2'])
        print(f"\nBest CTR Model: {best_ctr_model} (R² = {ctr_results[best_ctr_model]['r2']:.3f})")
        ctr_predictor.visualize_ctr_results(ctr_results[best_ctr_model])
    
    # 2. Retention Prediction
    print("\n2. Retention Prediction Analysis")
    retention_predictor = RetentionPredictor()
    
    # 尝试不同的模型
    retention_models = ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression']
    retention_results = {}
    
    for model_type in retention_models:
        try:
            result = retention_predictor.train_retention_model(df_combined, model_type=model_type)
            if result:
                retention_results[model_type] = result
                print(f"  {model_type.upper()}: Accuracy = {result['accuracy']:.3f}")
        except Exception as e:
            print(f"  {model_type.upper()}: Error - {e}")
    
    # 选择最佳留存模型
    if retention_results:
        best_retention_model = max(retention_results.keys(), key=lambda x: retention_results[x]['accuracy'])
        print(f"\nBest Retention Model: {best_retention_model} (Accuracy = {retention_results[best_retention_model]['accuracy']:.3f})")
        retention_predictor.visualize_retention_results(retention_results[best_retention_model])
    
    return {
        'ctr_results': ctr_results,
        'retention_results': retention_results,
        'ctr_predictor': ctr_predictor,
        'retention_predictor': retention_predictor
    }

if __name__ == "__main__":
    print("Prediction Models Module")
    print("This module provides CTR and retention prediction capabilities.")
    print("Use run_prediction_analysis() to run complete analysis.") 