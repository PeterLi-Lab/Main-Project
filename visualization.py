#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for Stack Overflow Data Analysis
Provides comprehensive plotting functions for metrics, model comparisons, and data exploration
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

class VisualizationModule:
    """Comprehensive visualization module for data analysis and model evaluation"""
    
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_roc_curves(self, results_dict, title="ROC Curves Comparison", save_path=None):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            if 'y_test' in results and 'y_pred' in results:
                y_test = results['y_test']
                y_pred = results['y_pred']
                
                fpr, tpr, _ = roc_curve(y_test, y_pred)
                auc_score = results.get('auc', 0)
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results_dict, metric='auc', title="Model Performance Comparison", save_path=None):
        """Plot bar chart comparing model performance"""
        plt.figure(figsize=(12, 8))
        
        models = list(results_dict.keys())
        metrics = [results_dict[model].get(metric, 0) for model in models]
        
        bars = plt.bar(models, metrics, color=sns.color_palette("husl", len(models)))
        plt.ylabel(metric.upper())
        plt.title(title)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, metric_val in zip(bars, metrics):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{metric_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_importance_df, top_n=20, title="Feature Importance", save_path=None):
        """Plot feature importance"""
        if feature_importance_df is None or len(feature_importance_df) == 0:
            print("No feature importance data available")
            return
        
        plt.figure(figsize=(12, 10))
        
        # Get top N features
        top_features = feature_importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_uplift_analysis(self, uplift_results, title="Uplift Analysis", save_path=None):
        """Plot uplift analysis results"""
        if not uplift_results:
            print("No uplift results available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Uplift distribution
        if 'treatment_pred' in uplift_results and 'control_pred' in uplift_results:
            treatment_pred = uplift_results['treatment_pred']
            control_pred = uplift_results['control_pred']
            uplift_dist = treatment_pred - control_pred
            
            axes[0, 0].hist(uplift_dist, bins=50, alpha=0.7)
            axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[0, 0].set_xlabel('Uplift (Treatment - Control)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Uplift Distribution')
        
        # 2. Click rate comparison
        if 'treatment_click_rate' in uplift_results and 'control_click_rate' in uplift_results:
            groups = ['Control', 'Treatment']
            rates = [uplift_results['control_click_rate'], uplift_results['treatment_click_rate']]
            
            bars = axes[0, 1].bar(groups, rates, color=['lightblue', 'lightgreen'])
            axes[0, 1].set_ylabel('Click Rate')
            axes[0, 1].set_title('Click Rate Comparison')
            
            for bar, rate in zip(bars, rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                               f'{rate:.3f}', ha='center', va='bottom')
        
        # 3. Prediction probability distribution
        if 'treatment_pred' in uplift_results and 'control_pred' in uplift_results:
            axes[1, 0].hist(control_pred, bins=30, alpha=0.7, label='Control', color='lightblue')
            axes[1, 0].hist(treatment_pred, bins=30, alpha=0.7, label='Treatment', color='lightgreen')
            axes[1, 0].set_xlabel('Predicted Click Probability')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Predicted Probability Distribution')
            axes[1, 0].legend()
        
        # 4. Uplift summary
        if 'uplift' in uplift_results:
            uplift_val = uplift_results['uplift']
            axes[1, 1].bar(['Uplift'], [uplift_val], color='orange')
            axes[1, 1].set_ylabel('Uplift Value')
            axes[1, 1].set_title('Uplift Summary')
            axes[1, 1].text(0, uplift_val + 0.001, f'{uplift_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_multitask_results(self, multitask_results, title="Multi-Task Learning Results", save_path=None):
        """Plot multi-task learning results"""
        if not multitask_results:
            print("No multi-task results available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Task performance comparison
        tasks = list(multitask_results.keys())
        auc_scores = [multitask_results[task]['auc'] for task in tasks]
        accuracy_scores = [multitask_results[task]['accuracy'] for task in tasks]
        
        x = np.arange(len(tasks))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, auc_scores, width, label='AUC', color='skyblue')
        axes[0, 0].bar(x + width/2, accuracy_scores, width, label='Accuracy', color='lightcoral')
        axes[0, 0].set_xlabel('Tasks')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Task Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(tasks)
        axes[0, 0].legend()
        
        # 2. ROC curves for each task
        for i, task in enumerate(tasks):
            if 'predictions' in multitask_results[task] and 'true_values' in multitask_results[task]:
                y_true = multitask_results[task]['true_values']
                y_pred = multitask_results[task]['predictions']
                
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                auc = multitask_results[task]['auc']
                
                axes[0, 1].plot(fpr, tpr, label=f'{task} (AUC={auc:.3f})', linewidth=2)
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Prediction distribution for each task
        for i, task in enumerate(tasks):
            if 'predictions' in multitask_results[task] and 'true_values' in multitask_results[task]:
                y_true = multitask_results[task]['true_values']
                y_pred = multitask_results[task]['predictions']
                
                axes[1, 0].scatter(y_true, y_pred, alpha=0.5, label=task)
        
        axes[1, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].set_title('Predicted vs Actual')
        axes[1, 0].legend()
        
        # 4. Metrics summary
        metrics_data = []
        for task in tasks:
            metrics_data.append({
                'Task': task,
                'AUC': multitask_results[task]['auc'],
                'Accuracy': multitask_results[task]['accuracy']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        axes[1, 1].table(cellText=metrics_df.values, colLabels=metrics_df.columns, 
                         cellLoc='center', loc='center')
        axes[1, 1].set_title('Metrics Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self, results_dict, save_dir=None):
        """Create a comprehensive visualization report"""
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Model comparison
        self.plot_model_comparison(results_dict, save_path=f"{save_dir}/model_comparison.png")
        
        # 2. ROC curves
        self.plot_roc_curves(results_dict, save_path=f"{save_dir}/roc_curves.png")
        
        print(f"Comprehensive report saved to: {save_dir}")
    
    def save_all_plots(self, results_dict, prefix="analysis"):
        """Save all plots with consistent naming"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create timestamped directory
        save_dir = f"{self.output_dir}/{prefix}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save all plots
        self.create_comprehensive_report(results_dict, save_dir)
        
        print(f"All plots saved to: {save_dir}")
        return save_dir

# Example usage
if __name__ == "__main__":
    viz = VisualizationModule()
    
    # Example data
    sample_results = {
        'Model1': {'auc': 0.85, 'accuracy': 0.82, 'y_test': [0,1,0,1], 'y_pred': [0.1,0.9,0.2,0.8]},
        'Model2': {'auc': 0.88, 'accuracy': 0.85, 'y_test': [0,1,0,1], 'y_pred': [0.2,0.8,0.1,0.9]}
    }
    
    viz.plot_model_comparison(sample_results)
    viz.plot_roc_curves(sample_results) 