import pandas as pd
import re

# 1. 读取聚类结果
in_csv = 'post_clusters.csv'
df = pd.read_csv(in_csv)

# 2. 筛选第7簇
cluster7 = df[df['cluster_id'] == 7].copy()
print(f"Cluster 7 post count: {len(cluster7):,}")

# 3. 定义AI相关关键词
ai_keywords = [
    r'ai', r'artificial intelligence', r'machine learning', r'deep learning', r'neural network',
    r'gpt', r'llm', r'data science', r'predictive', r'automated', r'intelligent', r'smart',
    r'tensorflow', r'pytorch', r'scikit-learn', r'openai', r'nlp', r'computer vision', r'reinforcement learning', r'transformer'
]
ai_pattern = re.compile('|'.join(ai_keywords), re.IGNORECASE)

# 4. 分组逻辑：只要merged_content中包含AI关键词即为treatment，否则为control
def assign_group(text):
    if pd.isnull(text):
        return 'control'
    return 'treatment' if ai_pattern.search(text) else 'control'

cluster7['group'] = cluster7['merged_content'].apply(assign_group)

# 5. 输出统计
n_treatment = (cluster7['group'] == 'treatment').sum()
n_control = (cluster7['group'] == 'control').sum()
print(f"Treatment: {n_treatment:,}  Control: {n_control:,}")

# 6. 导出分组结果
out_csv = 'cluster7_treatment_control.csv'
cluster7[['Id', 'Title', 'Body', 'Tags', 'merged_content', 'group']].to_csv(out_csv, index=False)
print(f"Exported: {out_csv}") 