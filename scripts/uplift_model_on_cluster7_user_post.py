import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. 读取cluster7分组信息
cluster7 = pd.read_csv('cluster7_treatment_control.csv')
cluster7_post_ids = set(cluster7['Id'].astype(str))
print(f"Cluster 7 post count: {len(cluster7_post_ids):,}")

# 2. 读取user-post点击样本，只保留cluster7的post
click_df = pd.read_csv('user_post_click_samples.csv')
click_df = click_df[click_df['post_id'].astype(str).isin(cluster7_post_ids)].copy()
print(f"User-post samples in cluster7: {len(click_df):,}")

# 3. 合并内容特征和treatment/control标签
# 先准备post内容特征和分组
post_info = cluster7.set_index(cluster7['Id'].astype(str))[['Title', 'Body', 'Tags', 'merged_content', 'group']]
click_df['post_id'] = click_df['post_id'].astype(str)
click_df = click_df.merge(post_info, left_on='post_id', right_index=True, how='left')

# 4. 特征工程（TF-IDF embedding + 简单统计特征）
print("Extracting features...")
vectorizer = TfidfVectorizer(max_features=200, stop_words='english', ngram_range=(1,2), min_df=2, max_df=0.95)
X_text = vectorizer.fit_transform(click_df['merged_content'].fillna(''))

click_df['body_length'] = click_df['Body'].fillna('').apply(len)
click_df['title_length'] = click_df['Title'].fillna('').apply(len)
click_df['num_tags'] = click_df['Tags'].fillna('').apply(lambda x: len(str(x).split('|')) if x else 0)

from scipy.sparse import hstack
X_stats = click_df[['body_length', 'title_length', 'num_tags']].values
X = hstack([X_text, X_stats])
y = click_df['is_click']
treatment = (click_df['group'] == 'treatment').astype(int)

# 5. Two-Model方法训练uplift（分别训练treatment/control模型）
print("Training uplift models (Two-Model approach)...")
indices = np.arange(len(click_df))
X_train, X_test, y_train, y_test, treat_train, treat_test, idx_train, idx_test = train_test_split(
    X, y, treatment, indices, test_size=0.3, random_state=42)

clf_treat = LogisticRegression(max_iter=200)
clf_treat.fit(X_train[treat_train==1], y_train[treat_train==1])
clf_ctrl = LogisticRegression(max_iter=200)
clf_ctrl.fit(X_train[treat_train==0], y_train[treat_train==0])

proba_treat = clf_treat.predict_proba(X_test)[:,1]
proba_ctrl = clf_ctrl.predict_proba(X_test)[:,1]
uplift = proba_treat - proba_ctrl

print(f"Mean predicted uplift: {uplift.mean():.4f}")
print(f"Top 5% uplift: {np.percentile(uplift, 95):.4f}")

plt.figure(figsize=(8,4))
plt.hist(uplift, bins=50, color='skyblue', edgecolor='k')
plt.title('Predicted Uplift Distribution (Cluster 7, User-Post)')
plt.xlabel('Predicted Uplift')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('uplift_predicted_distribution_cluster7_user_post.png')
print("Uplift distribution plot saved: uplift_predicted_distribution_cluster7_user_post.png")

click_df_test = click_df.iloc[idx_test].copy()
click_df_test['uplift_pred'] = uplift
out_csv = 'cluster7_user_post_uplift_prediction.csv'
click_df_test[['user_id', 'post_id', 'Title', 'Body', 'Tags', 'merged_content', 'group', 'is_click', 'uplift_pred']].to_csv(out_csv, index=False)
print(f"Uplift prediction exported: {out_csv}")

# 训练集/测试集分割后，分别评估两个模型
# Treatment模型评估
mask_treat = (treat_test == 1)
if mask_treat.sum() > 0:
    y_pred_treat = clf_treat.predict(X_test[mask_treat])
    acc_treat = accuracy_score(y_test[mask_treat], y_pred_treat)
    auc_treat = roc_auc_score(y_test[mask_treat], proba_treat[mask_treat])
    print(f"Treatment模型: 测试集准确率={acc_treat:.4f}  AUC={auc_treat:.4f}")
else:
    print("Treatment测试集无样本")
# Control模型评估
mask_ctrl = (treat_test == 0)
if mask_ctrl.sum() > 0:
    y_pred_ctrl = clf_ctrl.predict(X_test[mask_ctrl])
    acc_ctrl = accuracy_score(y_test[mask_ctrl], y_pred_ctrl)
    auc_ctrl = roc_auc_score(y_test[mask_ctrl], proba_ctrl[mask_ctrl])
    print(f"Control模型: 测试集准确率={acc_ctrl:.4f}  AUC={auc_ctrl:.4f}")
else:
    print("Control测试集无样本")

with open('uplift_model_eval.txt', 'w', encoding='utf-8') as f:
    f.write(f"Mean predicted uplift: {uplift.mean():.4f}\n")
    f.write(f"Top 5% uplift: {np.percentile(uplift, 95):.4f}\n")
    if mask_treat.sum() > 0:
        f.write(f"Treatment模型: 测试集准确率={acc_treat:.4f}  AUC={auc_treat:.4f}\n")
    else:
        f.write("Treatment测试集无样本\n")
    if mask_ctrl.sum() > 0:
        f.write(f"Control模型: 测试集准确率={acc_ctrl:.4f}  AUC={auc_ctrl:.4f}\n")
    else:
        f.write("Control测试集无样本\n") 