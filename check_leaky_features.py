import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_leaky_features():
    """逐个检查泄露特征"""
    print("=== 逐个检查泄露特征 ===\n")
    
    # 加载数据
    df = pd.read_csv('uplift_model_data.csv')
    
    # 泄露特征列表
    leaky_features = [
        'ai_interest_x_treatment',
        'user_ai_interest_score', 
        'user_previous_ai_click_rate',
        'user_ai_interest_weighted',
        'num_tags',
        'user_post_tag_overlap',
        'user_ai_interactions'
    ]
    
    print("=== 1. ai_interest_x_treatment ===")
    print(f"特征值范围: [{df['ai_interest_x_treatment'].min():.4f}, {df['ai_interest_x_treatment'].max():.4f}]")
    print(f"与 treatment 相关性: {abs(df['ai_interest_x_treatment'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['ai_interest_x_treatment']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    print(f"  唯一值数量: {control_data.nunique()}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['ai_interest_x_treatment']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    print(f"  唯一值数量: {treatment_data.nunique()}")
    
    # 检查这个特征是否直接等于 treatment
    print(f"\n特征值是否直接等于 treatment:")
    print(f"  完全相等: {(df['ai_interest_x_treatment'] == df['treatment_ai_content']).mean():.2%}")
    print(f"  高度相关: {(abs(df['ai_interest_x_treatment'] - df['treatment_ai_content']) < 0.1).mean():.2%}")
    
    print("\n=== 2. user_ai_interest_score ===")
    print(f"特征值范围: [{df['user_ai_interest_score'].min():.4f}, {df['user_ai_interest_score'].max():.4f}]")
    print(f"与 treatment 相关性: {abs(df['user_ai_interest_score'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['user_ai_interest_score']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['user_ai_interest_score']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    
    print("\n=== 3. user_previous_ai_click_rate ===")
    print(f"特征值范围: [{df['user_previous_ai_click_rate'].min():.4f}, {df['user_previous_ai_click_rate'].max():.4f}]")
    print(f"与 treatment 相关性: {abs(df['user_previous_ai_click_rate'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['user_previous_ai_click_rate']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['user_previous_ai_click_rate']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    
    print("\n=== 4. user_ai_interest_weighted ===")
    print(f"特征值范围: [{df['user_ai_interest_weighted'].min():.4f}, {df['user_ai_interest_weighted'].max():.4f}]")
    print(f"与 treatment 相关性: {abs(df['user_ai_interest_weighted'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['user_ai_interest_weighted']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['user_ai_interest_weighted']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    
    print("\n=== 5. num_tags ===")
    print(f"特征值范围: [{df['num_tags'].min():.0f}, {df['num_tags'].max():.0f}]")
    print(f"与 treatment 相关性: {abs(df['num_tags'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['num_tags']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['num_tags']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    
    print("\n=== 6. user_post_tag_overlap ===")
    print(f"特征值范围: [{df['user_post_tag_overlap'].min():.4f}, {df['user_post_tag_overlap'].max():.4f}]")
    print(f"与 treatment 相关性: {abs(df['user_post_tag_overlap'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['user_post_tag_overlap']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['user_post_tag_overlap']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    
    print("\n=== 7. user_ai_interactions ===")
    print(f"特征值范围: [{df['user_ai_interactions'].min():.4f}, {df['user_ai_interactions'].max():.4f}]")
    print(f"与 treatment 相关性: {abs(df['user_ai_interactions'].corr(df['treatment_ai_content'])):.4f}")
    
    # 检查这个特征在不同 treatment 下的分布
    print("\nTreatment=0 时的分布:")
    control_data = df[df['treatment_ai_content'] == 0]['user_ai_interactions']
    print(f"  均值: {control_data.mean():.4f}")
    print(f"  标准差: {control_data.std():.4f}")
    
    print("\nTreatment=1 时的分布:")
    treatment_data = df[df['treatment_ai_content'] == 1]['user_ai_interactions']
    print(f"  均值: {treatment_data.mean():.4f}")
    print(f"  标准差: {treatment_data.std():.4f}")
    
    print("\n=== 泄露原因分析 ===")
    print("1. ai_interest_x_treatment: 这个特征本身就是 treatment 的交互项，直接包含了 treatment 信息")
    print("2. user_ai_interest_score: 可能基于用户对 AI 内容的兴趣计算，与 treatment 高度相关")
    print("3. user_previous_ai_click_rate: 用户历史点击 AI 内容的比率，与 treatment 相关")
    print("4. user_ai_interest_weighted: 加权 AI 兴趣分数，与 treatment 相关")
    print("5. num_tags: 帖子标签数量，可能 AI 内容有特定的标签模式")
    print("6. user_post_tag_overlap: 用户与帖子标签重叠度，AI 内容可能有特定标签")
    print("7. user_ai_interactions: 用户与 AI 内容的交互历史，直接相关")
    
    print("\n=== 建议解决方案 ===")
    print("1. 移除所有包含 AI 相关信息的特征")
    print("2. 只使用用户和帖子的原始特征（如 user_reputation, post_length 等）")
    print("3. 确保特征在 treatment 分配之前就存在")
    print("4. 重新训练模型，使用干净的 feature set")
    
    return leaky_features

if __name__ == "__main__":
    leaky_features = check_leaky_features() 