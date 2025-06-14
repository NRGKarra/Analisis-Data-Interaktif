import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score,
                           confusion_matrix, classification_report)

class ModelEvaluator:
    def __init__(self):
        pass
    
    def classification_metrics(self, y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return metrics
    
    def regression_metrics(self, y_true, y_pred):
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }
        return metrics
    
    def get_confusion_matrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true, y_pred):
        return classification_report(y_true, y_pred, output_dict=True)
    
    def compare_models(self, models_results, metric='accuracy'):
        comparison = {}
        for model_name, results in models_results.items():
            if metric in results:
                comparison[model_name] = results[metric]
        if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'r2_score']:
            comparison = dict(sorted(comparison.items(), key=lambda x: x[1], reverse=True))
        else:
            comparison = dict(sorted(comparison.items(), key=lambda x: x[1]))
        return comparison
