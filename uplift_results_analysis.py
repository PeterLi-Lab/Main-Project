#!/usr/bin/env python3
"""
Uplift Model Results Analysis
Analyzes and visualizes uplift modeling results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class UpliftResultsAnalyzer:
    def __init__(self):
        """Initialize uplift results analyzer"""
        self.results = {
            'two_model': {
                'actual_uplift': -0.4304,
                'predicted_uplift': -0.3924,
                'uplift_error': 0.0379
            },
            'single_model': {
                'mse': 0.1663,
                'r2': 0.3326
            }
        }
        
    def analyze_treatment_effects(self):
        """Analyze treatment effects and business implications"""
        print("=== Uplift Model Results Analysis ===")
        print()
        
        # Treatment effect analysis
        actual_uplift = self.results['two_model']['actual_uplift']
        predicted_uplift = self.results['two_model']['predicted_uplift']
        error = self.results['two_model']['uplift_error']
        
        print("📊 TREATMENT EFFECT ANALYSIS")
        print("=" * 50)
        print(f"Actual Uplift: {actual_uplift:.4f}")
        print(f"Predicted Uplift: {predicted_uplift:.4f}")
        print(f"Prediction Error: {error:.4f}")
        print()
        
        # Business interpretation
        print("💼 BUSINESS INTERPRETATION")
        print("=" * 50)
        if actual_uplift < 0:
            print("❌ NEGATIVE TREATMENT EFFECT")
            print(f"   • AI-related tags REDUCE click-through rate by {abs(actual_uplift):.1%}")
            print("   • This suggests users are LESS likely to click on AI-tagged posts")
            print()
            print("🔍 POSSIBLE REASONS:")
            print("   • Users may be less interested in AI-related content")
            print("   • AI-tagged posts might be perceived as lower quality")
            print("   • Users prefer non-AI content")
            print()
            print("📈 RECOMMENDATIONS:")
            print("   • Avoid adding AI-related tags to posts")
            print("   • Consider removing AI tags from existing posts")
            print("   • Test alternative tag strategies")
            print("   • Investigate why AI content performs poorly")
        else:
            print("✅ POSITIVE TREATMENT EFFECT")
            print(f"   • AI-related tags INCREASE click-through rate by {actual_uplift:.1%}")
        
        print()
        
        # Model performance analysis
        print("🤖 MODEL PERFORMANCE ANALYSIS")
        print("=" * 50)
        print("Two-Model Approach (T-Learner):")
        print(f"   • Prediction Accuracy: {(1 - error/abs(actual_uplift))*100:.1f}%")
        print(f"   • Error Rate: {error:.4f}")
        print()
        print("Single Model with Treatment Interaction:")
        print(f"   • R² Score: {self.results['single_model']['r2']:.4f}")
        print(f"   • MSE: {self.results['single_model']['mse']:.4f}")
        print()
        
        if self.results['single_model']['r2'] > 0.3:
            print("✅ Good model performance - explains significant variance")
        else:
            print("⚠️  Model performance could be improved with better features")
        
        return True
    
    def create_visualizations(self):
        """Create visualizations for uplift results"""
        print("\n=== Creating Visualizations ===")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Uplift Model Results Analysis', fontsize=16, fontweight='bold')
        
        # 1. Uplift Comparison
        ax1 = axes[0, 0]
        uplift_values = [self.results['two_model']['actual_uplift'], 
                        self.results['two_model']['predicted_uplift']]
        labels = ['Actual Uplift', 'Predicted Uplift']
        colors = ['#ff6b6b', '#4ecdc4']
        
        bars = ax1.bar(labels, uplift_values, color=colors, alpha=0.7)
        ax1.set_title('Actual vs Predicted Uplift', fontweight='bold')
        ax1.set_ylabel('Uplift Value')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, uplift_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                    f'{value:.4f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 2. Prediction Error
        ax2 = axes[0, 1]
        error = self.results['two_model']['uplift_error']
        ax2.bar(['Prediction Error'], [error], color='#ffa726', alpha=0.7)
        ax2.set_title('Prediction Error', fontweight='bold')
        ax2.set_ylabel('Error Value')
        ax2.text(0, error/2, f'{error:.4f}', ha='center', va='center', fontweight='bold')
        
        # 3. Model Performance Metrics
        ax3 = axes[1, 0]
        metrics = ['R² Score', 'MSE']
        values = [self.results['single_model']['r2'], self.results['single_model']['mse']]
        colors_metrics = ['#66bb6a', '#ef5350']
        
        bars = ax3.bar(metrics, values, color=colors_metrics, alpha=0.7)
        ax3.set_title('Single Model Performance', fontweight='bold')
        ax3.set_ylabel('Metric Value')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 4. Treatment Effect Interpretation
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create text box with interpretation
        interpretation_text = f"""
Treatment Effect Analysis:

• Actual Uplift: {self.results['two_model']['actual_uplift']:.4f}
• Predicted Uplift: {self.results['two_model']['predicted_uplift']:.4f}
• Prediction Error: {self.results['two_model']['uplift_error']:.4f}

Business Impact:
• AI tags REDUCE click-through rate
• Users are less likely to click AI-tagged posts
• Consider removing AI tags to improve performance

Model Performance:
• T-Learner: {(1 - error/abs(self.results['two_model']['actual_uplift']))*100:.1f}% accuracy
• Single Model: R² = {self.results['single_model']['r2']:.4f}
        """
        
        ax4.text(0.05, 0.95, interpretation_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('uplift_results_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Saved visualization to 'uplift_results_analysis.png'")
        
        return True
    
    def generate_recommendations(self):
        """Generate actionable recommendations based on results"""
        print("\n=== ACTIONABLE RECOMMENDATIONS ===")
        print("=" * 50)
        
        actual_uplift = self.results['two_model']['actual_uplift']
        
        print("🎯 IMMEDIATE ACTIONS:")
        print("1. Remove AI-related tags from posts to improve click-through rates")
        print("2. A/B test alternative tag strategies")
        print("3. Investigate why AI content performs poorly")
        print()
        
        print("📊 FURTHER ANALYSIS NEEDED:")
        print("1. Segment analysis by user demographics")
        print("2. Content quality analysis of AI-tagged posts")
        print("3. User behavior analysis around AI content")
        print("4. Test different AI-related tag variations")
        print()
        
        print("🔧 MODEL IMPROVEMENTS:")
        print("1. Add more sophisticated features (user behavior, content quality)")
        print("2. Implement S-Learner and X-Learner approaches")
        print("3. Add propensity score matching")
        print("4. Include temporal features and user engagement history")
        print()
        
        print("📈 SUCCESS METRICS TO TRACK:")
        print("1. Click-through rate improvement after removing AI tags")
        print("2. User engagement metrics")
        print("3. Content quality scores")
        print("4. User satisfaction surveys")
        
        return True
    
    def run_full_analysis(self):
        """Run complete uplift results analysis"""
        print("=== Uplift Model Results Analysis ===")
        
        # Analyze treatment effects
        self.analyze_treatment_effects()
        
        # Create visualizations
        self.create_visualizations()
        
        # Generate recommendations
        self.generate_recommendations()
        
        print("\n✅ Analysis complete! Check 'uplift_results_analysis.png' for visualizations.")
        return True

def main():
    """Main function"""
    analyzer = UpliftResultsAnalyzer()
    analyzer.run_full_analysis()
    return analyzer

if __name__ == "__main__":
    main() 