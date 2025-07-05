#!/usr/bin/env python3
"""
Run Uplift Modeling on Real Data
"""

import os
import numpy as np
import pandas as pd
from data_preprocessing import DataPreprocessor
from prediction_models import UpliftModeling


def main():
    print("=== Real Data Uplift Modeling ===")
    # 1. 加载和预处理数据
    preprocessor = DataPreprocessor(base_path="data")
    print("1. 加载原始数据...")
    df_posts, df_users, df_tags, df_votes, df_badges = preprocessor.load_data()
    preprocessor.df_posts = df_posts
    preprocessor.df_users = df_users
    preprocessor.df_votes = df_votes
    preprocessor.df_badges = df_badges
    
    # 2. 生成uplift样本
    print("2. 生成uplift建模样本...")
    df_uplift = preprocessor.create_uplift_samples()
    if df_uplift is None or len(df_uplift) == 0:
        print("❌ 未能生成uplift样本，请检查原始数据！")
        return
    print(f"   ✓ 样本数: {len(df_uplift)}，Treatment组: {df_uplift['treatment'].sum()}，Control组: {(df_uplift['treatment']==0).sum()}")
    print(f"   ✓ 总点击率: {df_uplift['is_click'].mean():.3f}")
    
    # 3. 训练uplift模型
    print("3. 训练Uplift模型...")
    uplift_model = UpliftModeling()
    result = uplift_model.train_uplift_models(df_uplift, model_type='xgboost')
    if not result:
        print("❌ Uplift模型训练失败！")
        return
    print(f"   ✓ Treatment点击率: {result['treatment_click_rate']:.3f}")
    print(f"   ✓ Control点击率: {result['control_click_rate']:.3f}")
    print(f"   ✓ Uplift提升: {result['uplift']:.3f} ({result['uplift']/result['control_click_rate']*100:.1f}%)")
    
    # 4. 预测新样本
    print("4. 预测前10条样本的uplift...")
    new_data = df_uplift.head(10).copy()
    new_data_prepared, _ = uplift_model.prepare_uplift_features(new_data)
    uplift_scores, treatment_probs, control_probs = uplift_model.predict_uplift(new_data_prepared)
    print("   Uplift分数:")
    for i, score in enumerate(uplift_scores):
        print(f"    样本{i+1}: uplift={score:.3f}, treatment_prob={treatment_probs[i]:.3f}, control_prob={control_probs[i]:.3f}")
    
    # 5. 特征重要性
    if hasattr(uplift_model.treatment_model, 'feature_importances_'):
        print("\n5. 主要特征重要性:")
        feature_importance = pd.DataFrame({
            'feature': uplift_model.feature_names,
            'importance': uplift_model.treatment_model.feature_importances_
        }).sort_values('importance', ascending=False)
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
    print("\n=== 运行完成 ===")

if __name__ == "__main__":
    main() 