from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np


class TrafficMetrics:
    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.true_counts = []
        self.predicted_counts = []

    def add_classification_result(self, y_true, y_pred):
        self.true_labels.append(y_true)
        self.predicted_labels.append(y_pred)

    def add_regression_result(self, y_true, y_pred):
        self.true_counts.append(y_true)
        self.predicted_counts.append(y_pred)

    def get_classification_metrics(self):
        if not self.true_labels:
            return {}

        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predicted_labels)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': [[tn, fp], [fn, tp]],
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_negatives': tn,
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

    def get_regression_metrics(self):
        if not self.true_counts:
            return {}

        y_true = np.array(self.true_counts)
        y_pred = np.array(self.predicted_counts)

        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'error_distribution': (y_pred - y_true).tolist()
        }

    def print_all_metrics(self):
        cls_metrics = self.get_classification_metrics()
        reg_metrics = self.get_regression_metrics()

        print("\n=== Classification Metrics ===")
        if cls_metrics:
            print(f"Accuracy: {cls_metrics['accuracy']:.4f}")
            print(f"Precision: {cls_metrics['precision']:.4f}")
            print(f"Recall: {cls_metrics['recall']:.4f}")
            print(f"F1 Score: {cls_metrics['f1']:.4f}")
            print("\nConfusion Matrix:")
            print(f"True Negatives: {cls_metrics['true_negatives']}")
            print(f"False Positives: {cls_metrics['false_positives']}")
            print(f"False Negatives: {cls_metrics['false_negatives']}")
            print(f"True Positives: {cls_metrics['true_positives']}")

        print("\n=== Regression Metrics ===")
        if reg_metrics:
            print(f"MAE: {reg_metrics['mae']:.2f}")
            print(f"MSE: {reg_metrics['mse']:.2f}")
            print(f"RMSE: {reg_metrics['rmse']:.2f}")
            print(f"R²: {reg_metrics['r2']:.4f}")

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
#     classification_report
#
#
# class TrafficMetrics:
#     @staticmethod
#     def calculate_metrics(y_true, y_pred):
#         """Calculate evaluation metrics"""
#         return {
#             'accuracy': accuracy_score(y_true, y_pred),
#             'precision': precision_score(y_true, y_pred),
#             'recall': recall_score(y_true, y_pred),
#             'f1': f1_score(y_true, y_pred),
#             'confusion_matrix': confusion_matrix(y_true, y_pred),
#             'report': classification_report(y_true, y_pred)
#         }
#
#     @staticmethod
#     def print_metrics(metrics):
#         """Print formatted metrics"""
#         print("\n=== Model Evaluation Metrics ===")
#         print(f"Accuracy: {metrics['accuracy']:.4f}")
#         print(f"Precision: {metrics['precision']:.4f}")
#         print(f"Recall: {metrics['recall']:.4f}")
#         print(f"F1 Score: {metrics['f1']:.4f}")
#         print("\nConfusion Matrix:")
#         print(metrics['confusion_matrix'])
#         print("\nClassification Report:")
#         print(metrics['report'])