import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def clean_uplift_analysis():
    """使用干净特征进行 uplift 分析"""
    print("=== 干净特征 Uplift 分析 ===\n")
    
    # 加载数据
    df = pd.read_csv('uplift_model_data.csv')
    print(f"总数据量: {len(df):,}")
    
    # 定义干净特征（移除所有泄露特征）
    clean_features = [
        'user_reputation', 'user_post_count', 'user_account_age_days',
        'total_badges', 'gold_badges', 'silver_badges', 'bronze_badges',
        'unique_badge_types', 'badge_rate_per_day', 'recent_badges_30d',
        'badge_quality_score', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount',
        'title_length', 'post_length', 'post_age_days', 'total_votes', 'upvotes',
        'content_quality_score', 'engagement_rate', 'content_complexity'
    ]
    
    # 只保留存在的特征
    available_clean_features = [col for col in clean_features if col in df.columns]
    print(f"干净特征数量: {len(available_clean_features)}")
    print("干净特征列表:")
    for feature in available_clean_features:
        print(f"  - {feature}")
    
    # 准备数据
    X = df[available_clean_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # 移除 NaN 值
    valid_mask = ~response.isna()
    X = X[valid_mask]
    treatment = treatment[valid_mask]
    response = response[valid_mask]
    
    print(f"\n有效样本数: {len(X):,}")
    
    # 检查干净特征与 treatment 的相关性
    print("\n=== 干净特征与 Treatment 相关性检查 ===")
    treatment_correlations = []
    for col in available_clean_features:
        if X[col].dtype in ['int64', 'float64']:
            corr = abs(X[col].corr(treatment))
            treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 与 treatment 相关性最高的干净特征:")
    for col, corr in treatment_correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # 检查是否有高相关性特征
    high_corr_features = [col for col, corr in treatment_correlations if corr > 0.3]
    print(f"\n与 treatment 相关性 > 0.3 的干净特征数量: {len(high_corr_features)}")
    if high_corr_features:
        print("⚠️  警告: 仍有高相关性特征!")
        for col in high_corr_features:
            print(f"  - {col}")
    else:
        print("✅ 干净特征集没有高相关性特征")
    
    # 使用 50% 测试集进行严格验证
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.5, random_state=42, stratify=treatment
    )
    
    print(f"\n训练集: {len(X_train):,} 样本")
    print(f"测试集: {len(X_test):,} 样本")
    
    # 计算实际 uplift
    actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print(f"\n实际 Uplift: {actual_uplift:.4f}")
    
    # 训练模型
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    print(f"\n训练 treatment 样本: {len(X_treatment):,}")
    print(f"训练 control 样本: {len(X_control):,}")
    
    # 训练 XGBoost 模型
    print("\n=== 训练 XGBoost 模型 ===")
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
    
    # 预测
    y_pred_treatment = treatment_model.predict(X_test.values)
    y_pred_control = control_model.predict(X_test.values)
    
    # 计算 uplift 预测
    uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
    uplift_error = abs(actual_uplift - uplift_pred)
    uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
    
    print(f"\n=== 干净特征结果 ===")
    print(f"实际 Uplift: {actual_uplift:.4f}")
    print(f"预测 Uplift: {uplift_pred:.4f}")
    print(f"Uplift Error: {uplift_error:.4f}")
    print(f"Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # 交叉验证
    cv_scores_treatment = cross_val_score(treatment_model, X_treatment.values, y_treatment.values, cv=5, scoring='r2')
    cv_scores_control = cross_val_score(control_model, X_control.values, y_control.values, cv=5, scoring='r2')
    
    print(f"\n交叉验证结果:")
    print(f"Treatment Model CV R²: {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
    print(f"Control Model CV R²: {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
    
    # 特征重要性
    print(f"\n=== 特征重要性 (Treatment Model) ===")
    importance_df = pd.DataFrame({
        'feature': available_clean_features,
        'importance': treatment_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 特征重要性:")
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 预测分布检查
    print(f"\n=== 预测分布检查 ===")
    print(f"Treatment 预测均值: {y_pred_treatment.mean():.4f}")
    print(f"Control 预测均值: {y_pred_control.mean():.4f}")
    print(f"Treatment 预测方差: {y_pred_treatment.var():.4f}")
    print(f"Control 预测方差: {y_pred_control.var():.4f}")
    
    # 结论
    print(f"\n=== 结论 ===")
    if uplift_accuracy > 0.8:
        print("✅ 干净特征集达到了合理的准确率")
    elif uplift_accuracy > 0.5:
        print("⚠️  准确率一般，可能需要更多特征工程")
    else:
        print("❌ 准确率较低，可能需要重新设计特征")
    
    print(f"Uplift 预测方向: {'正确' if (actual_uplift > 0 and uplift_pred > 0) or (actual_uplift < 0 and uplift_pred < 0) else '错误'}")
    
    return {
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_pred,
        'uplift_error': uplift_error,
        'uplift_accuracy': uplift_accuracy,
        'clean_features': available_clean_features
    }

if __name__ == "__main__":
    results = clean_uplift_analysis() 