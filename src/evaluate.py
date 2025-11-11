import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_pred_proba=None):
        """
        Compute evaluation metrics
        
        Args:
            y_true: True labels (0: normal, 1: attack)
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            
        Returns:
            metrics: Dictionary of metrics
        """
        # Convert OC-SVM output (-1, 1) to (1, 0)
        if np.min(y_pred) == -1:
            y_pred = np.where(y_pred == 1, 0, 1)
        
        # Compute metrics
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        self.metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        })
        
        # ROC-AUC if probabilities provided
        if y_pred_proba is not None:
            self.metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return self.metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        print(f"\nFalse Positive Rate: {self.metrics['false_positive_rate']:.4f}")
        print(f"False Negative Rate: {self.metrics['false_negative_rate']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"\nROC-AUC Score: {self.metrics['roc_auc']:.4f}")
        
        print("\n" + "="*50)
        print("CONFUSION MATRIX")
        print("="*50)
        print(f"True Negatives:  {self.metrics['true_negatives']}")
        print(f"False Positives: {self.metrics['false_positives']}")
        print(f"False Negatives: {self.metrics['false_negatives']}")
        print(f"True Positives:  {self.metrics['true_positives']}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        # Convert predictions if needed
        if np.min(y_pred) == -1:
            y_pred = np.where(y_pred == 1, 0, 1)
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0.5, 1.5], ['Normal', 'Attack'])
        plt.yticks([0.5, 1.5], ['Normal', 'Attack'])
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()