import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def validate_uplift_results():
    """严格验证 uplift 建模结果"""
    print("=== 严格验证 Uplift 建模结果 ===\n")
    
    # 加载数据
    df = pd.read_csv('uplift_model_data.csv')
    print(f"总数据量: {len(df):,}")
    
    # 检查数据分布
    print("\n=== 数据分布检查 ===")
    print(f"Treatment 分布:")
    print(f"  Control (0): {(df['treatment_ai_content'] == 0).sum():,} ({(df['treatment_ai_content'] == 0).mean():.2%})")
    print(f"  Treatment (1): {(df['treatment_ai_content'] == 1).sum():,} ({(df['treatment_ai_content'] == 1).mean():.2%})")
    
    print(f"\nResponse 分布:")
    print(f"  No click (0): {(df['response'] == 0).sum():,} ({(df['response'] == 0).mean():.2%})")
    print(f"  Click (1): {(df['response'] == 1).sum():,} ({(df['response'] == 1).mean():.2%})")
    
    # 检查特征与 treatment 的相关性
    print("\n=== 特征与 Treatment 相关性检查 ===")
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    
    treatment_correlations = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            corr = abs(df[col].corr(df['treatment_ai_content']))
            treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 与 treatment 相关性最高的特征:")
    for col, corr in treatment_correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # 检查数据泄露
    print("\n=== 数据泄露检查 ===")
    high_corr_features = [col for col, corr in treatment_correlations if corr > 0.5]
    print(f"与 treatment 相关性 > 0.5 的特征数量: {len(high_corr_features)}")
    if high_corr_features:
        print("⚠️  警告: 可能存在数据泄露!")
        for col in high_corr_features:
            print(f"  - {col}")
    
    # 使用更严格的验证方法
    print("\n=== 严格验证方法 ===")
    
    # 1. 更大的测试集
    X = df[feature_cols].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # 使用 50% 测试集
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.5, random_state=42, stratify=treatment
    )
    
    print(f"训练集: {len(X_train):,} 样本")
    print(f"测试集: {len(X_test):,} 样本")
    
    # 2. 计算实际 uplift
    actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print(f"\n实际 Uplift: {actual_uplift:.4f}")
    
    # 3. 训练模型并预测
    treatment_mask_train = t_train == 1
    control_mask_train = t_train == 0
    
    X_treatment = X_train[treatment_mask_train]
    y_treatment = y_train[treatment_mask_train]
    X_control = X_train[control_mask_train]
    y_control = y_train[control_mask_train]
    
    # 训练模型
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
    
    print(f"\n=== 严格验证结果 ===")
    print(f"实际 Uplift: {actual_uplift:.4f}")
    print(f"预测 Uplift: {uplift_pred:.4f}")
    print(f"Uplift Error: {uplift_error:.4f}")
    print(f"Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # 4. 交叉验证
    print(f"\n=== 交叉验证结果 ===")
    cv_scores_treatment = cross_val_score(treatment_model, X_treatment.values, y_treatment.values, cv=5, scoring='r2')
    cv_scores_control = cross_val_score(control_model, X_control.values, y_control.values, cv=5, scoring='r2')
    
    print(f"Treatment Model CV R²: {cv_scores_treatment.mean():.4f} ± {cv_scores_treatment.std():.4f}")
    print(f"Control Model CV R²: {cv_scores_control.mean():.4f} ± {cv_scores_control.std():.4f}")
    
    # 5. 检查预测分布
    print(f"\n=== 预测分布检查 ===")
    print(f"Treatment 预测均值: {y_pred_treatment.mean():.4f}")
    print(f"Control 预测均值: {y_pred_control.mean():.4f}")
    print(f"Treatment 预测方差: {y_pred_treatment.var():.4f}")
    print(f"Control 预测方差: {y_pred_control.var():.4f}")
    
    # 6. 检查是否预测值都在合理范围内
    print(f"\n=== 预测值范围检查 ===")
    print(f"Treatment 预测范围: [{y_pred_treatment.min():.4f}, {y_pred_treatment.max():.4f}]")
    print(f"Control 预测范围: [{y_pred_control.min():.4f}, {y_pred_control.max():.4f}]")
    
    # 检查是否所有预测值都在 [0,1] 范围内
    treatment_in_range = np.all((y_pred_treatment >= 0) & (y_pred_treatment <= 1))
    control_in_range = np.all((y_pred_control >= 0) & (y_pred_control <= 1))
    
    print(f"Treatment 预测值在 [0,1] 范围内: {treatment_in_range}")
    print(f"Control 预测值在 [0,1] 范围内: {control_in_range}")
    
    # 7. 结论
    print(f"\n=== 验证结论 ===")
    if uplift_accuracy > 0.95:
        print("⚠️  警告: 准确率过高，可能存在以下问题:")
        print("  1. 数据泄露 - 特征中包含了 treatment 信息")
        print("  2. 过拟合 - 模型过于复杂")
        print("  3. 测试集太小 - 结果不稳定")
        print("  4. 特征工程问题 - 某些特征直接预测了目标")
    else:
        print("✅ 准确率合理，模型可信")
    
    if len(high_corr_features) > 0:
        print("⚠️  发现与 treatment 高度相关的特征，可能存在数据泄露")
    
    return {
        'actual_uplift': actual_uplift,
        'predicted_uplift': uplift_pred,
        'uplift_error': uplift_error,
        'uplift_accuracy': uplift_accuracy,
        'high_corr_features': high_corr_features
    }

if __name__ == "__main__":
    results = validate_uplift_results() 