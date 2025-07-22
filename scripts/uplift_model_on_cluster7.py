import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 1. 读取数据
df = pd.read_csv('cluster7_treatment_control.csv')
print(f"Loaded {len(df):,} samples from cluster7_treatment_control.csv")

# 2. 合并真实click作为response
click_df = pd.read_csv('user_post_click_samples.csv', usecols=['post_id', 'is_click'])
click_df = click_df.groupby('post_id')['is_click'].max().reset_index()  # 只要有一次点击就算1
click_df.rename(columns={'post_id': 'Id', 'is_click': 'response'}, inplace=True)
df['Id'] = pd.to_numeric(df['Id'], errors='coerce')
df = df.merge(click_df, on='Id', how='left')
df['response'] = df['response'].fillna(0).astype(int)
print(f"Response positive rate: {df['response'].mean():.3f}")

# 3. 特征工程（TF-IDF embedding + 简单统计特征）
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
X_text = vectorizer.fit_transform(df['merged_content'])

# 统计特征
df['body_length'] = df['Body'].fillna('').apply(len)
df['title_length'] = df['Title'].fillna('').apply(len)
df['num_tags'] = df['Tags'].fillna('').apply(lambda x: len(str(x).split('|')) if x else 0)

X_stats = df[['body_length', 'title_length', 'num_tags']].values

# 合并特征
from scipy.sparse import hstack
X = hstack([X_text, X_stats])

y = df['response']
treatment = (df['group'] == 'treatment').astype(int)

# 4. Two-Model方法训练uplift（分别训练treatment/control模型）
print("Training uplift models (Two-Model approach)...")
# 保存索引
indices = np.arange(len(df))
X_train, X_test, y_train, y_test, treat_train, treat_test, idx_train, idx_test = train_test_split(
    X, y, treatment, indices, test_size=0.3, random_state=42)

# Treatment模型
clf_treat = LogisticRegression(max_iter=200)
clf_treat.fit(X_train[treat_train==1], y_train[treat_train==1])
# Control模型
clf_ctrl = LogisticRegression(max_iter=200)
clf_ctrl.fit(X_train[treat_train==0], y_train[treat_train==0])

# 预测uplift
proba_treat = clf_treat.predict_proba(X_test)[:,1]
proba_ctrl = clf_ctrl.predict_proba(X_test)[:,1]
uplift = proba_treat - proba_ctrl

# 5. 评估uplift效果
print(f"Mean predicted uplift: {uplift.mean():.4f}")
print(f"Top 5% uplift: {np.percentile(uplift, 95):.4f}")

# 6. 可视化uplift分布
plt.figure(figsize=(8,4))
plt.hist(uplift, bins=50, color='skyblue', edgecolor='k')
plt.title('Predicted Uplift Distribution (Cluster 7, Real Click)')
plt.xlabel('Predicted Uplift')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('uplift_predicted_distribution_cluster7_real_click.png')
print("Uplift distribution plot saved: uplift_predicted_distribution_cluster7_real_click.png")

# 7. 导出预测结果
# 修正：用idx_test索引df，确保一一对应
df_test = df.iloc[idx_test].copy()
df_test['uplift_pred'] = uplift
out_csv = 'cluster7_uplift_prediction_real_click.csv'
df_test[['Id', 'Title', 'Body', 'Tags', 'merged_content', 'group', 'response', 'uplift_pred']].to_csv(out_csv, index=False)
print(f"Uplift prediction exported: {out_csv}") 