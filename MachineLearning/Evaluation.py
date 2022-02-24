import numpy as np
from sklearn import metrics
from typing import Dict, List, Iterable


def root_mean_squared_error(y_true: Iterable, y_pred: Iterable):
    return metrics.mean_squared_error(y_true, y_pred, squared=False)


class Evaluation:
    cls_metrics = {'accuracy': metrics.accuracy_score, 'precision': metrics.precision_score,
                   'recall': metrics.recall_score, 'f1': metrics.f1_score, 'balanced': metrics.balanced_accuracy_score,
                   'kappa': metrics.cohen_kappa_score, 'auc': metrics.roc_auc_score}
    reg_metrics = {'mae': metrics.mean_absolute_error, 'mse': metrics.mean_squared_error, 'max': metrics.max_error,
                   'rmse': root_mean_squared_error, 'variance': metrics.explained_variance_score,
                   'mape': metrics.mean_absolute_percentage_error}

    @staticmethod
    def multi_classification(stats: List[str], y_true: Iterable, y_pred: Iterable) -> Dict[str, float]:
        # Calculate metrics for known stats and save them in dictionary
        return {s: Evaluation.cls_metrics[s](y_true, y_pred) for s in stats if s in Evaluation.cls_metrics.keys()}

    @staticmethod
    def single_classification(stat: str, y_true: Iterable, y_pred: Iterable) -> float:
        if stat not in Evaluation.cls_metrics.keys():
            raise ValueError(f'Unknown classification metric {stat}!')
        return Evaluation.cls_metrics[stat](y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true: Iterable, y_pred: Iterable, normalize: str) -> np.ndarray:
        return metrics.confusion_matrix(y_true, y_pred, normalize=normalize)

    @staticmethod
    def regression_evaluation(stats: List[str], y_true: Iterable, y_pred: Iterable) -> Dict[str, float]:
        # Calculate metrics for known stats and save them in dictionary
        return {s: Evaluation.reg_metrics[s](y_true, y_pred) for s in stats if s in Evaluation.reg_metrics.keys()}

    @staticmethod
    def single_regression(stat: str, y_true: Iterable, y_pred: Iterable) -> float:
        if stat not in Evaluation.reg_metrics.keys():
            raise ValueError(f'Unknown regression metric {stat}!')
        return Evaluation.reg_metrics[stat](y_true, y_pred)
