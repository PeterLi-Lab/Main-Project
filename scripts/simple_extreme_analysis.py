import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def main():
    """Simple analysis of uplift distribution and extreme values"""
    
    print("=== SIMPLE EXTREME VALUE ANALYSIS ===")
    
    # Load the existing results
    try:
        df = pd.read_csv('simple_uplift_results.csv')
        print(f"Loaded {len(df):,} posts with uplift scores")
    except FileNotFoundError:
        print("simple_uplift_results.csv not found. Running basic uplift analysis first...")
        # Run the basic analysis first
        import subprocess
        subprocess.run(['python', 'scripts/simple_uplift_analysis.py'])
        df = pd.read_csv('simple_uplift_results.csv')
    
    # Get uplift scores
    uplift_scores = df['uplift_score'].values
    
    print(f"\n=== UPLIFT DISTRIBUTION ANALYSIS ===")
    print(f"Mean: {uplift_scores.mean():.6f}")
    print(f"Median: {np.median(uplift_scores):.6f}")
    print(f"Std: {uplift_scores.std():.6f}")
    print(f"Min: {uplift_scores.min():.6f}")
    print(f"Max: {uplift_scores.max():.6f}")
    
    # Distribution shape
    print(f"\nDistribution shape:")
    print(f"Skewness: {pd.Series(uplift_scores).skew():.3f}")
    print(f"Kurtosis: {pd.Series(uplift_scores).kurtosis():.3f}")
    
    # Check for bimodal pattern
    print(f"\nBimodal analysis:")
    print(f"Positive uplift: {(uplift_scores > 0).sum():,} ({((uplift_scores > 0).sum()/len(uplift_scores)*100):.1f}%)")
    print(f"Negative uplift: {(uplift_scores < 0).sum():,} ({((uplift_scores < 0).sum()/len(uplift_scores)*100):.1f}%)")
    print(f"Zero uplift: {(uplift_scores == 0).sum():,} ({((uplift_scores == 0).sum()/len(uplift_scores)*100):.1f}%)")
    
    # Percentile analysis
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentile analysis:")
    for p in percentiles:
        value = np.percentile(uplift_scores, p)
        print(f"{p}th percentile: {value:.6f}")
    
    # Find extreme values
    print(f"\n=== EXTREME VALUES ANALYSIS ===")
    top_1_percent = np.percentile(uplift_scores, 99)
    bottom_1_percent = np.percentile(uplift_scores, 1)
    
    print(f"Top 1% threshold: {top_1_percent:.6f}")
    print(f"Bottom 1% threshold: {bottom_1_percent:.6f}")
    
    # Get extreme posts
    top_posts = df[uplift_scores >= top_1_percent]
    bottom_posts = df[uplift_scores <= bottom_1_percent]
    
    print(f"\nTop 1% posts: {len(top_posts):,}")
    print(f"Bottom 1% posts: {len(bottom_posts):,}")
    
    # Analyze characteristics
    print(f"\n=== TOP 1% POSTS CHARACTERISTICS ===")
    print(f"Treatment ratio: {(top_posts['treatment'] == 1).mean():.3f}")
    print(f"Average click rate: {top_posts['click_rate'].mean():.3f}")
    print(f"Average title length: {top_posts['title_length'].mean():.1f}")
    print(f"Average AI keyword count: {top_posts['ai_keyword_count'].mean():.1f}")
    
    print(f"\n=== BOTTOM 1% POSTS CHARACTERISTICS ===")
    print(f"Treatment ratio: {(bottom_posts['treatment'] == 1).mean():.3f}")
    print(f"Average click rate: {bottom_posts['click_rate'].mean():.3f}")
    print(f"Average title length: {bottom_posts['title_length'].mean():.1f}")
    print(f"Average AI keyword count: {bottom_posts['ai_keyword_count'].mean():.1f}")
    
    # Show sample posts
    print(f"\n=== SAMPLE TOP UPLIFT POSTS ===")
    for i, (_, row) in enumerate(top_posts.head(3).iterrows()):
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"{i+1}. Uplift: {row['uplift_score']:.6f}, Treatment: {row['treatment']}, Click Rate: {row['click_rate']:.3f}")
        print(f"   Title: {title[:80]}...")
        print(f"   Tags: {row['Tags']}")
        print()
    
    print(f"\n=== SAMPLE BOTTOM UPLIFT POSTS ===")
    for i, (_, row) in enumerate(bottom_posts.head(3).iterrows()):
        title = str(row['Title']) if pd.notna(row['Title']) else "No title"
        print(f"{i+1}. Uplift: {row['uplift_score']:.6f}, Treatment: {row['treatment']}, Click Rate: {row['click_rate']:.3f}")
        print(f"   Title: {title[:80]}...")
        print(f"   Tags: {row['Tags']}")
        print()
    
    # Subgroup analysis
    print(f"\n=== SUBGROUP ANALYSIS ===")
    
    # By treatment group
    treatment_uplift = uplift_scores[df['treatment'] == 1]
    control_uplift = uplift_scores[df['treatment'] == 0]
    
    print(f"Treatment group uplift: mean={treatment_uplift.mean():.6f}, std={treatment_uplift.std():.6f}")
    print(f"Control group uplift: mean={control_uplift.mean():.6f}, std={control_uplift.std():.6f}")
    
    # By click rate level
    high_click = df[df['click_rate'] > df['click_rate'].median()]
    low_click = df[df['click_rate'] <= df['click_rate'].median()]
    
    print(f"High click rate uplift: mean={high_click['uplift_score'].mean():.6f}")
    print(f"Low click rate uplift: mean={low_click['uplift_score'].mean():.6f}")
    
    # Create simple visualization
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(uplift_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(uplift_scores.mean(), color='red', linestyle='--', label=f'Mean: {uplift_scores.mean():.6f}')
    plt.xlabel('Uplift Score')
    plt.ylabel('Frequency')
    plt.title('Uplift Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(2, 2, 2)
    plt.boxplot(uplift_scores)
    plt.ylabel('Uplift Score')
    plt.title('Uplift Score Box Plot')
    plt.grid(True, alpha=0.3)
    
    # Treatment vs Control
    plt.subplot(2, 2, 3)
    plt.hist(treatment_uplift, bins=30, alpha=0.7, label='Treatment', color='lightgreen', density=True)
    plt.hist(control_uplift, bins=30, alpha=0.7, label='Control', color='lightcoral', density=True)
    plt.xlabel('Uplift Score')
    plt.ylabel('Density')
    plt.title('Uplift Distribution by Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 2, 4)
    plt.scatter(df['click_rate'], df['uplift_score'], alpha=0.5, s=1)
    plt.xlabel('Click Rate')
    plt.ylabel('Uplift Score')
    plt.title('Uplift Score vs Click Rate')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('extreme_value_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print("Visualization saved as 'extreme_value_analysis.png'")

if __name__ == "__main__":
    main()







