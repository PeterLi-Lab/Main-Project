import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def final_validation_check():
    """最终验证检查，识别其他可能导致高准确率的问题"""
    print("=== Final Validation Check ===\n")
    
    # 加载数据
    df = pd.read_csv('uplift_model_data.csv')
    print(f"总数据量: {len(df):,}")
    
    # 1. 检查数据分布是否过于简单
    print("\n=== 1. 数据分布检查 ===")
    
    # 检查response的分布
    response_dist = df['response'].value_counts(normalize=True)
    print(f"Response 分布:")
    for value, ratio in response_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # 检查treatment的分布
    treatment_dist = df['treatment_ai_content'].value_counts(normalize=True)
    print(f"\nTreatment 分布:")
    for value, ratio in treatment_dist.items():
        print(f"  {value}: {ratio:.2%}")
    
    # 检查是否数据过于不平衡
    if response_dist.min() < 0.01:  # 如果某个类别占比小于1%
        print("⚠️  数据严重不平衡，可能导致模型过拟合")
    
    # 2. 检查特征是否过于简单
    print("\n=== 2. 特征复杂度检查 ===")
    
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    # 检查特征的唯一值数量
    simple_features = []
    for col in numeric_features:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.01:  # 如果唯一值比例小于1%
            simple_features.append((col, unique_ratio))
    
    print(f"过于简单的特征 (唯一值比例 < 1%):")
    for col, ratio in simple_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 3. 检查是否存在确定性关系
    print("\n=== 3. 确定性关系检查 ===")
    
    # 检查是否有特征与response有完美的相关性
    perfect_corr_features = []
    for col in numeric_features:
        corr = abs(df[col].corr(df['response']))
        if corr > 0.95:  # 如果相关性超过95%
            perfect_corr_features.append((col, corr))
    
    print(f"与response几乎完美相关的特征 (>95%):")
    for col, corr in perfect_corr_features:
        print(f"  {col}: {corr:.4f}")
    
    # 4. 检查模型复杂度
    print("\n=== 4. 模型复杂度检查 ===")
    
    # 使用不同的模型复杂度进行测试
    model_configs = [
        {'name': 'Simple', 'n_estimators': 10, 'max_depth': 2},
        {'name': 'Medium', 'n_estimators': 50, 'max_depth': 4},
        {'name': 'Complex', 'n_estimators': 100, 'max_depth': 8},
        {'name': 'Very Complex', 'n_estimators': 200, 'max_depth': 12}
    ]
    
    X = df[numeric_features].fillna(0)
    treatment = df['treatment_ai_content']
    response = df['response']
    
    # 分割数据
    X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
        X, treatment, response, test_size=0.3, random_state=42, stratify=treatment
    )
    
    complexity_results = []
    
    for config in model_configs:
        print(f"\n测试 {config['name']} 模型:")
        
        # 训练模型
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
        treatment_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            random_state=42, verbosity=0
        )
        control_model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'], 
            max_depth=config['max_depth'],
            random_state=42, verbosity=0
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
        
        complexity_results.append({
            'name': config['name'],
            'uplift_accuracy': uplift_accuracy,
            'actual_uplift': actual_uplift,
            'predicted_uplift': uplift_pred
        })
        
        print(f"  Uplift Accuracy: {uplift_accuracy:.2%}")
    
    # 检查复杂度对准确率的影响
    print(f"\n模型复杂度对准确率的影响:")
    for result in complexity_results:
        print(f"  {result['name']}: {result['uplift_accuracy']:.2%}")
    
    # 5. 检查数据泄露的其他形式
    print("\n=== 5. 其他数据泄露检查 ===")
    
    # 检查是否有特征直接等于response
    direct_response_features = []
    for col in numeric_features:
        if df[col].equals(df['response']):
            direct_response_features.append(col)
    
    print(f"直接等于response的特征: {direct_response_features}")
    
    # 检查是否有特征是response的线性变换
    linear_response_features = []
    for col in numeric_features:
        if col != 'response':
            # 检查是否与response有线性关系
            corr = abs(df[col].corr(df['response']))
            if corr > 0.99:  # 如果相关性超过99%
                linear_response_features.append((col, corr))
    
    print(f"与response几乎线性相关的特征 (>99%):")
    for col, corr in linear_response_features:
        print(f"  {col}: {corr:.4f}")
    
    # 6. 检查特征选择问题
    print("\n=== 6. 特征选择问题检查 ===")
    
    # 检查是否有重复特征
    duplicate_features = []
    for i, col1 in enumerate(numeric_features):
        for j, col2 in enumerate(numeric_features[i+1:], i+1):
            if df[col1].equals(df[col2]):
                duplicate_features.append((col1, col2))
    
    print(f"完全重复的特征对:")
    for col1, col2 in duplicate_features:
        print(f"  {col1} = {col2}")
    
    # 检查是否有高度相关的特征
    high_corr_pairs = []
    for i, col1 in enumerate(numeric_features):
        for j, col2 in enumerate(numeric_features[i+1:], i+1):
            corr = abs(df[col1].corr(df[col2]))
            if corr > 0.95:
                high_corr_pairs.append((col1, col2, corr))
    
    print(f"高度相关的特征对 (>95%):")
    for col1, col2, corr in high_corr_pairs:
        print(f"  {col1} <-> {col2}: {corr:.4f}")
    
    # 7. 检查数据质量问题
    print("\n=== 7. 数据质量问题检查 ===")
    
    # 检查缺失值
    missing_info = df[numeric_features].isnull().sum()
    high_missing = missing_info[missing_info > 0]
    print(f"有缺失值的特征:")
    for col, missing in high_missing.items():
        print(f"  {col}: {missing:,} ({missing/len(df):.2%})")
    
    # 检查异常值
    outlier_features = []
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_ratio = outliers / len(df)
        if outlier_ratio > 0.2:  # 如果异常值超过20%
            outlier_features.append((col, outlier_ratio))
    
    print(f"\n异常值比例高的特征 (>20%):")
    for col, ratio in outlier_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 8. 检查随机性
    print("\n=== 8. 随机性检查 ===")
    
    # 使用不同的随机种子测试
    seeds = [42, 123, 456, 789, 999]
    random_results = []
    
    for seed in seeds:
        X_train, X_test, t_train, t_test, y_train, y_test = train_test_split(
            X, treatment, response, test_size=0.3, random_state=seed, stratify=treatment
        )
        
        treatment_mask_train = t_train == 1
        control_mask_train = t_train == 0
        
        X_treatment = X_train[treatment_mask_train]
        y_treatment = y_train[treatment_mask_train]
        X_control = X_train[control_mask_train]
        y_control = y_train[control_mask_train]
        
        treatment_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=seed, verbosity=0)
        control_model = xgb.XGBRegressor(n_estimators=50, max_depth=4, random_state=seed, verbosity=0)
        
        treatment_model.fit(X_treatment.values, y_treatment.values)
        control_model.fit(X_control.values, y_control.values)
        
        y_pred_treatment = treatment_model.predict(X_test.values)
        y_pred_control = control_model.predict(X_test.values)
        
        actual_uplift = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
        uplift_pred = y_pred_treatment[t_test == 1].mean() - y_pred_control[t_test == 0].mean()
        uplift_error = abs(actual_uplift - uplift_pred)
        uplift_accuracy = max(0, 1 - uplift_error / abs(actual_uplift)) if actual_uplift != 0 else 0
        
        random_results.append(uplift_accuracy)
    
    print(f"不同随机种子的准确率:")
    for i, accuracy in enumerate(random_results):
        print(f"  Seed {seeds[i]}: {accuracy:.2%}")
    
    accuracy_variance = np.var(random_results)
    print(f"准确率方差: {accuracy_variance:.4f}")
    
    if accuracy_variance < 0.001:
        print("⚠️  准确率过于稳定，可能存在确定性关系")
    
    # 9. 最终结论
    print("\n=== 9. 最终结论 ===")
    
    issues = []
    
    if len(perfect_corr_features) > 0:
        issues.append("发现与response几乎完美相关的特征")
    
    if len(direct_response_features) > 0:
        issues.append("发现直接等于response的特征")
    
    if len(linear_response_features) > 0:
        issues.append("发现与response几乎线性相关的特征")
    
    if len(duplicate_features) > 0:
        issues.append("发现重复特征")
    
    if accuracy_variance < 0.001:
        issues.append("准确率过于稳定，可能存在确定性关系")
    
    if issues:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\n建议:")
        print("  1. 移除与response高度相关的特征")
        print("  2. 移除重复特征")
        print("  3. 检查数据预处理步骤")
        print("  4. 重新设计特征工程")
    else:
        print("✅ 未发现明显的确定性关系问题")
    
    return {
        'perfect_corr_features': perfect_corr_features,
        'direct_response_features': direct_response_features,
        'linear_response_features': linear_response_features,
        'duplicate_features': duplicate_features,
        'high_corr_pairs': high_corr_pairs,
        'complexity_results': complexity_results,
        'random_results': random_results,
        'issues': issues
    }

if __name__ == "__main__":
    results = final_validation_check() 