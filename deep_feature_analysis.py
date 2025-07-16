import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

def deep_feature_analysis():
    """深度分析可能导致问题的特征"""
    print("=== Deep Feature Analysis ===\n")
    
    # 加载数据
    df = pd.read_csv('uplift_model_data.csv')
    print(f"总数据量: {len(df):,}")
    
    # 1. 详细分析高相关性特征
    print("\n=== 1. 高相关性特征详细分析 ===")
    
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response']]
    numeric_features = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    # 计算相关性
    treatment_correlations = []
    response_correlations = []
    
    for col in numeric_features:
        treatment_corr = abs(df[col].corr(df['treatment_ai_content']))
        response_corr = abs(df[col].corr(df['response']))
        treatment_correlations.append((col, treatment_corr))
        response_correlations.append((col, response_corr))
    
    treatment_correlations.sort(key=lambda x: x[1], reverse=True)
    response_correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 分析前10个高相关性特征
    high_corr_features = [col for col, corr in treatment_correlations[:10]]
    
    print("前10个与treatment相关性最高的特征:")
    for i, (col, corr) in enumerate(treatment_correlations[:10]):
        print(f"{i+1}. {col}: {corr:.4f}")
        
        # 分析每个特征的分布
        treatment_group = df[df['treatment_ai_content'] == 1][col]
        control_group = df[df['treatment_ai_content'] == 0][col]
        
        print(f"   Treatment组: 均值={treatment_group.mean():.4f}, 标准差={treatment_group.std():.4f}")
        print(f"   Control组: 均值={control_group.mean():.4f}, 标准差={control_group.std():.4f}")
        print(f"   差异: {treatment_group.mean() - control_group.mean():.4f}")
        print()
    
    # 2. 检查特征是否直接包含treatment信息
    print("\n=== 2. Treatment信息泄露检查 ===")
    
    # 检查ai_interest_x_treatment特征
    if 'ai_interest_x_treatment' in df.columns:
        print("分析 ai_interest_x_treatment 特征:")
        ai_interest = df['user_ai_interest_score']
        treatment = df['treatment_ai_content']
        interaction = df['ai_interest_x_treatment']
        
        print(f"  user_ai_interest_score 范围: [{ai_interest.min():.4f}, {ai_interest.max():.4f}]")
        print(f"  treatment 值: {treatment.unique()}")
        print(f"  ai_interest_x_treatment 范围: [{interaction.min():.4f}, {interaction.max():.4f}]")
        
        # 验证交互特征的计算
        expected_interaction = ai_interest * treatment
        is_correct = np.allclose(interaction, expected_interaction, rtol=1e-10)
        print(f"  交互特征计算是否正确: {is_correct}")
        
        if is_correct:
            print("  ⚠️  这个特征直接包含了treatment信息，会导致数据泄露!")
    
    # 3. 分析用户AI相关特征
    print("\n=== 3. 用户AI相关特征分析 ===")
    
    ai_related_features = [col for col in numeric_features if 'ai' in col.lower()]
    print(f"AI相关特征: {ai_related_features}")
    
    for col in ai_related_features:
        print(f"\n分析 {col}:")
        
        # 按treatment分组分析
        treatment_group = df[df['treatment_ai_content'] == 1][col]
        control_group = df[df['treatment_ai_content'] == 0][col]
        
        print(f"  Treatment组统计:")
        print(f"    均值: {treatment_group.mean():.4f}")
        print(f"    中位数: {treatment_group.median():.4f}")
        print(f"    标准差: {treatment_group.std():.4f}")
        print(f"    最小值: {treatment_group.min():.4f}")
        print(f"    最大值: {treatment_group.max():.4f}")
        
        print(f"  Control组统计:")
        print(f"    均值: {control_group.mean():.4f}")
        print(f"    中位数: {control_group.median():.4f}")
        print(f"    标准差: {control_group.std():.4f}")
        print(f"    最小值: {control_group.min():.4f}")
        print(f"    最大值: {control_group.max():.4f}")
        
        # 检查是否有明显的分布差异
        mean_diff = treatment_group.mean() - control_group.mean()
        print(f"  均值差异: {mean_diff:.4f}")
        
        if abs(mean_diff) > 0.1:
            print(f"  ⚠️  该特征在treatment和control组间有明显差异，可能存在数据泄露")
    
    # 4. 检查特征的时间顺序
    print("\n=== 4. 特征时间顺序检查 ===")
    
    # 检查是否有基于未来信息的特征
    future_features = []
    for col in numeric_features:
        if 'previous' in col.lower() or 'history' in col.lower():
            future_features.append(col)
    
    print(f"包含历史信息的特征: {future_features}")
    
    for col in future_features:
        print(f"\n分析 {col}:")
        
        # 检查这个特征是否包含了treatment后的信息
        treatment_group = df[df['treatment_ai_content'] == 1][col]
        control_group = df[df['treatment_ai_content'] == 0][col]
        
        # 如果treatment组的特征值明显高于control组，可能存在数据泄露
        mean_diff = treatment_group.mean() - control_group.mean()
        print(f"  Treatment vs Control 均值差异: {mean_diff:.4f}")
        
        if mean_diff > 0.1:
            print(f"  ⚠️  该特征可能包含了treatment后的信息")
    
    # 5. 检查特征的多重共线性
    print("\n=== 5. 多重共线性检查 ===")
    
    # 选择相关性最高的特征
    top_features = [col for col, corr in treatment_correlations[:15]]
    
    # 计算特征间的相关性矩阵
    feature_matrix = df[top_features].fillna(0)
    correlation_matrix = feature_matrix.corr()
    
    # 找出高度相关的特征对
    high_corr_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            corr = abs(correlation_matrix.iloc[i, j])
            if corr > 0.8:
                high_corr_pairs.append((top_features[i], top_features[j], corr))
    
    print(f"高度相关的特征对 (|相关系数| > 0.8):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  {feat1} <-> {feat2}: {corr:.4f}")
    
    # 6. 检查特征的可解释性
    print("\n=== 6. 特征可解释性检查 ===")
    
    # 检查是否有过于复杂的特征
    complex_features = []
    for col in numeric_features:
        # 检查特征值的分布
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:  # 如果几乎每个值都不同
            complex_features.append((col, unique_ratio))
    
    print(f"过于复杂的特征 (唯一值比例 > 90%):")
    for col, ratio in complex_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 7. 检查特征的数据质量问题
    print("\n=== 7. 数据质量问题检查 ===")
    
    # 检查异常值
    outlier_features = []
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_ratio = outliers / len(df)
        if outlier_ratio > 0.1:  # 如果异常值超过10%
            outlier_features.append((col, outlier_ratio))
    
    print(f"异常值比例高的特征 (>10%):")
    for col, ratio in outlier_features:
        print(f"  {col}: {ratio:.2%}")
    
    # 8. 建议和结论
    print("\n=== 8. 建议和结论 ===")
    
    issues = []
    
    # 检查数据泄露
    if 'ai_interest_x_treatment' in df.columns:
        issues.append("ai_interest_x_treatment 特征直接包含treatment信息")
    
    high_corr_treatment = [col for col, corr in treatment_correlations if corr > 0.5]
    if len(high_corr_treatment) > 0:
        issues.append(f"发现 {len(high_corr_treatment)} 个与treatment高度相关的特征")
    
    if len(high_corr_pairs) > 0:
        issues.append(f"发现 {len(high_corr_pairs)} 对高度相关的特征，存在多重共线性")
    
    if len(complex_features) > 0:
        issues.append(f"发现 {len(complex_features)} 个过于复杂的特征")
    
    if len(outlier_features) > 0:
        issues.append(f"发现 {len(outlier_features)} 个异常值比例高的特征")
    
    if issues:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\n建议的解决方案:")
        print("  1. 移除 ai_interest_x_treatment 特征")
        print("  2. 移除与treatment高度相关的用户AI特征")
        print("  3. 处理多重共线性问题")
        print("  4. 处理异常值")
    else:
        print("✅ 未发现明显的数据质量问题")
    
    return {
        'high_corr_features': high_corr_features,
        'high_corr_pairs': high_corr_pairs,
        'complex_features': complex_features,
        'outlier_features': outlier_features,
        'issues': issues
    }

if __name__ == "__main__":
    results = deep_feature_analysis() 