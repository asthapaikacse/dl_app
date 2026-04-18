"""
Cost-sensitive learning and threshold optimization
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            roc_curve, precision_recall_curve)
import pandas as pd

class CostSensitiveLoss(nn.Module):
    """
    Method 5: Cost-Sensitive Loss
    Penalize false negatives (missing stroke) more heavily
    """
    def __init__(self, class_weights=[1.0, 1.63], fn_cost=5.0):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.fn_cost = fn_cost  # Cost of false negative
        
    def forward(self, inputs, targets):
        # Standard weighted CE
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights.to(inputs.device), 
                                  reduction='none')
        
        # Additional penalty for false negatives (predicting Normal when it's Stroke)
        probs = F.softmax(inputs, dim=1)
        fn_penalty = (targets == 1).float() * (probs[:, 0]) * self.fn_cost
        
        loss = ce_loss + fn_penalty
        return loss.mean()

class FocalLoss(nn.Module):
    """
    Focus on hard examples (usually minority class)
    """
    def __init__(self, alpha=0.75, gamma=2.0, class_weights=[1.0, 1.63]):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = torch.tensor(class_weights)
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights.to(inputs.device),
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def find_optimal_threshold(y_true, y_prob, target_recall=0.95):
    """
    Method 3: Find threshold that achieves target recall (sensitivity)
    Critical for medical diagnosis - don't miss strokes
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    # Find threshold that gives at least target recall
    idx = np.where(tpr >= target_recall)[0]
    if len(idx) > 0:
        # Among those, pick one with best precision (lowest FPR)
        optimal_idx = idx[np.argmin(fpr[idx])]
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    return optimal_threshold

class MetricsTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_specificity': []
        }
        self.best_metrics = {}
        self.optimal_threshold = 0.5
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)
                
    def plot_training_history(self, save_path):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Val', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['train_acc'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history['val_acc'], label='Val', linewidth=2)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(self.history['val_f1'], color='green', linewidth=2)
        axes[0, 2].set_title('Validation F1-Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['val_precision'], label='Precision', linewidth=2)
        axes[1, 0].plot(self.history['val_recall'], label='Recall/Sensitivity', linewidth=2)
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        if self.history['val_specificity']:
            axes[1, 1].plot(self.history['val_specificity'], color='purple', linewidth=2)
            axes[1, 1].set_title('Specificity')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.model_name} Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = 0.0
            
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=['Normal', 'Stroke']):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 16, 'weight': 'bold'})
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    textstr = '\n'.join((
        f'Sensitivity (Recall): {sensitivity:.3f}',
        f'Specificity: {specificity:.3f}',
        f'PPV (Precision): {ppv:.3f}',
        f'NPV: {npv:.3f}'
    ))
    
    plt.figtext(0.02, 0.02, textstr, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return cm

def plot_roc_curve(y_true, y_prob, save_path, model_name='Model'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    # Find optimal threshold (Youden's index)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100, 
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_threshold

def save_predictions_to_csv(file_paths, y_true, y_pred, y_prob, save_path, threshold=0.5):
    results_df = pd.DataFrame({
        'file_path': file_paths,
        'actual_label': ['Stroke' if y == 1 else 'Normal' for y in y_true],
        'predicted_label': ['Stroke' if y == 1 else 'Normal' for y in y_pred],
        'stroke_probability': y_prob,
        'prediction_correct': y_true == y_pred,
        'threshold_used': threshold
    })
    results_df.to_csv(save_path, index=False)
    print(f"Predictions saved: {save_path}")