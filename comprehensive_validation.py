import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def comprehensive_validation():
    """全面验证 uplift 建模结果，检查所有潜在问题"""
    print("=== Comprehensive Uplift Modeling Validation ===\n")
    
    # 加载数据
    df = pd.read_csv('uplift_model_data.csv')
    print(f"总数据量: {len(df):,}")
    
    # 1. 基础数据检查
    print("\n=== 1. 基础数据检查 ===")
    print(f"Treatment 分布:")
    print(f"  Control (0): {(df['treatment_ai_content'] == 0).sum():,} ({(df['treatment_ai_content'] == 0).mean():.2%})")
    print(f"  Treatment (1): {(df['treatment_ai_content'] == 1).sum():,} ({(df['treatment_ai_content'] == 1).mean():.2%})")
    
    print(f"\nResponse 分布:")
    print(f"  No click (0): {(df['response'] == 0).sum():,} ({(df['response'] == 0).mean():.2%})")
    print(f"  Click (1): {(df['response'] == 1).sum():,} ({(df['response'] == 1).mean():.2%})")
    
    # 2. 特征分析
    print("\n=== 2. 特征分析 ===")
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    print(f"特征数量: {len(feature_cols)}")
    
    # 检查特征类型
    numeric_features = []
    categorical_features = []
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"数值特征: {len(numeric_features)}")
    print(f"分类特征: {len(categorical_features)}")
    
    # 3. 数据泄露检查
    print("\n=== 3. 数据泄露检查 ===")
    
    # 检查与 treatment 的相关性
    treatment_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['treatment_ai_content']))
        treatment_correlations.append((col, corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    print("Top 15 与 treatment 相关性最高的特征:")
    for col, corr in treatment_correlations[:15]:
        print(f"  {col}: {corr:.4f}")
    
    # 检查与 response 的相关性
    response_correlations = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['response']))
        response_correlations.append((col, corr))
    
    response_correlations.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 15 与 response 相关性最高的特征:")
    for col, corr in response_correlations[:15]:
        print(f"  {col}: {corr:.4f}")
    
    # 4. 特征工程问题检查
    print("\n=== 4. 特征工程问题检查 ===")
    
    # 检查是否有直接包含 treatment 信息的特征
    treatment_leaky_features = []
    for col in feature_cols:
        if 'treatment' in col.lower() or 'ai_content' in col.lower():
            treatment_leaky_features.append(col)
    
    print(f"可能包含 treatment 信息的特征: {len(treatment_leaky_features)}")
    for col in treatment_leaky_features:
        print(f"  - {col}")
    
    # 检查是否有交互特征
    interaction_features = [col for col in feature_cols if 'x_' in col or '_x_' in col]
    print(f"\n交互特征数量: {len(interaction_features)}")
    for col in interaction_features[:10]:  # 只显示前10个
        print(f"  - {col}")
    
    # 5. 过拟合检查
    print("\n=== 5. 过拟合检查 ===")
    
    # 使用不同的测试集大小
    test_sizes = [0.2, 0.3, 0.5, 0.7]
    overfitting_results = []
    
    for test_size in test_sizes:
        X = df[numeric_features].fillna(0)
        treatment = df['treatment_ai_content']
        response = df['response']
        
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=test_size, random_state=42, stratify=treatment
        )
        
        # 训练模型
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
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
        
        # 计算 uplift
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        overfitting_results.append({
            'test_size': test_size,
            'train_size': len(X_train),
            'test_size_actual': len(X_test),
            'uplift_accuracy': uplift_accuracy,
            'actual_uplift': actual_uplift,
            'predicted_uplift': uplift_pred
        })
    
    print("不同测试集大小的准确率:")
    for result in overfitting_results:
        print(f"  测试集 {result['test_size']:.1%} ({result['test_size_actual']:,} 样本): {result['uplift_accuracy']:.2%}")
    
    # 6. 交叉验证
    print("\n=== 6. 交叉验证 ===")
    
    X = df[numeric_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    treatment_mask = treatment == 1
    control_mask = treatment == 0
    
    X_treatment = X[treatment_mask]
    y_treatment = response[treatment_mask]
    X_control = X[control_mask]
    y_control = response[control_mask]
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    treatment_cv_scores = []
    control_cv_scores = []
    
    for train_idx, val_idx in kf.split(X_treatment):
        X_train_fold = X_treatment.iloc[train_idx]
        y_train_fold = y_treatment.iloc[train_idx]
        X_val_fold = X_treatment.iloc[val_idx]
        y_val_fold = y_treatment.iloc[val_idx]
        
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        model.fit(X_train_fold.values, y_train_fold.values)
        y_pred = model.predict(X_val_fold.values)
        score = r2_score(y_val_fold.values, y_pred)
        treatment_cv_scores.append(score)
    
    for train_idx, val_idx in kf.split(X_control):
        X_train_fold = X_control.iloc[train_idx]
        y_train_fold = y_control.iloc[train_idx]
        X_val_fold = X_control.iloc[val_idx]
        y_val_fold = y_control.iloc[val_idx]
        
        model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
        model.fit(X_train_fold.values, y_train_fold.values)
        y_pred = model.predict(X_val_fold.values)
        score = r2_score(y_val_fold.values, y_pred)
        control_cv_scores.append(score)
    
    print(f"Treatment Model CV R²: {np.mean(treatment_cv_scores):.4f} ± {np.std(treatment_cv_scores):.4f}")
    print(f"Control Model CV R²: {np.mean(control_cv_scores):.4f} ± {np.std(control_cv_scores):.4f}")
    
    # 7. 特征重要性分析
    print("\n=== 7. 特征重要性分析 ===")
    
    # 训练完整模型
    treatment_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    control_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=42, verbosity=0)
    
    treatment_model.fit(X_treatment.values, y_treatment.values)
    control_model.fit(X_control.values, y_control.values)
    
    # 获取特征重要性
    treatment_importance = treatment_model.feature_importances_
    control_importance = control_model.feature_importances_
    
    # 排序
    treatment_feature_importance = list(zip(numeric_features, treatment_importance))
    control_feature_importance = list(zip(numeric_features, control_importance))
    
    treatment_feature_importance.sort(key=lambda x: x[1], reverse=True)
    control_feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("Treatment Model Top 10 重要特征:")
    for feature, importance in treatment_feature_importance[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nControl Model Top 10 重要特征:")
    for feature, importance in control_feature_importance[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    # 8. 数据质量问题
    print("\n=== 8. 数据质量问题 ===")
    
    # 检查缺失值
    missing_values = df[numeric_features].isnull().sum()
    high_missing = missing_values[missing_values > 0]
    print(f"有缺失值的特征数量: {len(high_missing)}")
    if len(high_missing) > 0:
        print("缺失值最多的特征:")
        for col, missing in high_missing.nlargest(5).items():
            print(f"  {col}: {missing:,} ({missing/len(df):.2%})")
    
    # 检查异常值
    print(f"\n异常值检查 (使用 IQR 方法):")
    outlier_features = []
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_features.append((col, outliers))
    
    outlier_features.sort(key=lambda x: x[1], reverse=True)
    print(f"有异常值的特征数量: {len(outlier_features)}")
    if outlier_features:
        print("异常值最多的特征:")
        for col, outliers in outlier_features[:5]:
            print(f"  {col}: {outliers:,} ({outliers/len(df):.2%})")
    
    # 9. 结论和建议
    print("\n=== 9. 结论和建议 ===")
    
    # 检查高相关性特征
    high_corr_treatment = [col for col, corr in treatment_correlations if corr > 0.3]
    high_corr_response = [col for col, corr in response_correlations if corr > 0.5]
    
    issues_found = []
    
    if len(high_corr_treatment) > 0:
        issues_found.append(f"发现 {len(high_corr_treatment)} 个与 treatment 高度相关的特征")
    
    if len(treatment_leaky_features) > 0:
        issues_found.append(f"发现 {len(treatment_leaky_features)} 个可能包含 treatment 信息的特征")
    
    if len(interaction_features) > 20:
        issues_found.append(f"交互特征过多 ({len(interaction_features)} 个)")
    
    # 检查过拟合
    accuracy_variance = np.var([r['uplift_accuracy'] for r in overfitting_results])
    if accuracy_variance > 0.01:
        issues_found.append("不同测试集大小的准确率差异较大，可能存在过拟合")
    
    if issues_found:
        print("⚠️  发现以下潜在问题:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("✅ 未发现明显问题")
    
    # 建议
    print("\n建议:")
    if len(high_corr_treatment) > 0:
        print("  1. 移除与 treatment 高度相关的特征")
    if len(treatment_leaky_features) > 0:
        print("  2. 检查并移除包含 treatment 信息的特征")
    if len(interaction_features) > 20:
        print("  3. 减少交互特征数量，避免过拟合")
    
    return {
        'treatment_correlations': treatment_correlations,
        'response_correlations': response_correlations,
        'treatment_leaky_features': treatment_leaky_features,
        'interaction_features': interaction_features,
        'overfitting_results': overfitting_results,
        'treatment_cv_scores': treatment_cv_scores,
        'control_cv_scores': control_cv_scores,
        'issues_found': issues_found
    }

if __name__ == "__main__":
    results = comprehensive_validation() 