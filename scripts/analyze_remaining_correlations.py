import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def analyze_remaining_correlations():
    """Analyze remaining correlations after initial feature selection"""
    print("=== Remaining Correlations Analysis ===\n")
    
    # Load data
    df = pd.read_csv('uplift_model_data.csv')
    print(f"Total data volume: {len(df):,}")
    
    # Check data structure
    print(f"Data columns: {list(df.columns)}")
    print(f"Data shape: {df.shape}")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['treatment_ai_content', 'response', 'user_id', 'post_id']]
    
    # Check feature types
    numeric_features = []
    categorical_features = []
    
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
        else:
            categorical_features.append(col)
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Handle missing values
    df = df.fillna(0)
    
    # 1. Calculate correlation matrix
    print("\n=== 1. Correlation Matrix Analysis ===")
    
    if len(numeric_features) > 1:
        correlation_matrix = df[numeric_features].corr()
        
        print(f"Correlation matrix shape: {correlation_matrix.shape}")
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(numeric_features)):
            for j in range(i+1, len(numeric_features)):
                corr = abs(correlation_matrix.iloc[i, j])
                if corr > 0.8:
                    high_corr_pairs.append((numeric_features[i], numeric_features[j], corr))
        
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nHighly correlated feature pairs (|correlation| > 0.8):")
        if high_corr_pairs:
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  {feat1} <-> {feat2}: {corr:.4f}")
        else:
            print("  No highly correlated pairs found")
        
        # Find moderately correlated feature pairs
        moderate_corr_pairs = []
        for i in range(len(numeric_features)):
            for j in range(i+1, len(numeric_features)):
                corr = abs(correlation_matrix.iloc[i, j])
                if 0.5 < corr <= 0.8:
                    moderate_corr_pairs.append((numeric_features[i], numeric_features[j], corr))
        
        moderate_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\nModerately correlated feature pairs (0.5 < |correlation| ≤ 0.8):")
        if moderate_corr_pairs:
            for feat1, feat2, corr in moderate_corr_pairs[:10]:  # Show top 10
                print(f"  {feat1} <-> {feat2}: {corr:.4f}")
        else:
            print("  No moderately correlated pairs found")
    
    # 2. Analyze correlations with target variables
    print("\n=== 2. Target Variable Correlations ===")
    
    if 'response' in df.columns:
        # Correlations with response
        response_correlations = []
        for col in numeric_features:
            corr = abs(df[col].corr(df['response']))
            response_correlations.append((col, corr))
        
        response_correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 15 features correlated with response:")
        for col, corr in response_correlations[:15]:
            print(f"  {col}: {corr:.4f}")
    
    if 'treatment_ai_content' in df.columns:
        # Correlations with treatment
        treatment_correlations = []
        for col in numeric_features:
            corr = abs(df[col].corr(df['treatment_ai_content']))
            treatment_correlations.append((col, corr))
        
        treatment_correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 15 features correlated with treatment:")
        for col, corr in treatment_correlations[:15]:
            print(f"  {col}: {corr:.4f}")
    
    # 3. Analyze feature clusters
    print("\n=== 3. Feature Cluster Analysis ===")
    
    if len(numeric_features) > 10:
        # Select top features by response correlation
        top_features = [col for col, corr in response_correlations[:20]]
        
        # Calculate correlation matrix for top features
        top_correlation_matrix = df[top_features].corr()
        
        # Find feature clusters
        clusters = []
        used_features = set()
        
        for i, feat1 in enumerate(top_features):
            if feat1 in used_features:
                continue
            
            cluster = [feat1]
            used_features.add(feat1)
            
            for j, feat2 in enumerate(top_features[i+1:], i+1):
                if feat2 in used_features:
                    continue
                
                corr = abs(top_correlation_matrix.iloc[i, j])
                if corr > 0.7:
                    cluster.append(feat2)
                    used_features.add(feat2)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        print(f"Feature clusters (correlation > 0.7):")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {cluster}")
    
    # 4. Analyze correlation patterns
    print("\n=== 4. Correlation Pattern Analysis ===")
    
    # Check for correlation patterns by feature type
    feature_types = {}
    for col in numeric_features:
        # Categorize features by name patterns
        if 'user' in col.lower():
            feature_types[col] = 'user'
        elif 'post' in col.lower():
            feature_types[col] = 'post'
        elif 'interaction' in col.lower():
            feature_types[col] = 'interaction'
        elif 'time' in col.lower() or 'date' in col.lower():
            feature_types[col] = 'time'
        else:
            feature_types[col] = 'other'
    
    # Analyze correlations within feature types
    for feature_type in ['user', 'post', 'interaction', 'time', 'other']:
        type_features = [col for col, ftype in feature_types.items() if ftype == feature_type]
        
        if len(type_features) > 1:
            print(f"\n{feature_type.capitalize()} features ({len(type_features)} features):")
            
            # Calculate average correlation within type
            type_correlations = []
            for i, feat1 in enumerate(type_features):
                for j, feat2 in enumerate(type_features[i+1:], i+1):
                    corr = abs(df[feat1].corr(df[feat2]))
                    type_correlations.append(corr)
            
            if type_correlations:
                avg_corr = np.mean(type_correlations)
                max_corr = max(type_correlations)
                print(f"  Average correlation: {avg_corr:.4f}")
                print(f"  Max correlation: {max_corr:.4f}")
                
                if avg_corr > 0.5:
                    print(f"  ⚠️  High average correlation within {feature_type} features")
    
    # 5. Summary and recommendations
    print("\n=== 5. Summary and Recommendations ===")
    
    issues = []
    recommendations = []
    
    if len(high_corr_pairs) > 0:
        issues.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        recommendations.append("Consider removing one feature from each highly correlated pair")
    
    if len(moderate_corr_pairs) > 10:
        issues.append(f"Found {len(moderate_corr_pairs)} moderately correlated feature pairs")
        recommendations.append("Consider feature selection to reduce multicollinearity")
    
    if len(clusters) > 0:
        issues.append(f"Found {len(clusters)} feature clusters")
        recommendations.append("Consider using one representative feature from each cluster")
    
    if issues:
        print("⚠️  Found the following correlation issues:")
        for issue in issues:
            print(f"  - {issue}")
        
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("✅ No significant correlation issues found")
    
    return {
        'high_corr_pairs': high_corr_pairs,
        'moderate_corr_pairs': moderate_corr_pairs,
        'response_correlations': response_correlations if 'response' in df.columns else None,
        'treatment_correlations': treatment_correlations if 'treatment_ai_content' in df.columns else None,
        'clusters': clusters,
        'issues': issues,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = analyze_remaining_correlations() 