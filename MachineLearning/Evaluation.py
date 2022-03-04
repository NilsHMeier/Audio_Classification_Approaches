import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.metrics import Metric
from typing import Dict, List, Iterable


def root_mean_squared_error(y_true: Iterable, y_pred: Iterable):
    return metrics.mean_squared_error(y_true, y_pred, squared=False)


def prediction_error(y_true: Iterable, y_pred: Iterable, normalize: bool = False):
    cm = metrics.confusion_matrix(y_true, y_pred)
    true_count = np.dot(np.sum(cm, axis=1), np.arange(len(cm)))
    pred_count = np.dot(np.sum(cm, axis=0), np.arange(len(cm)))
    return np.abs(true_count - pred_count) / true_count if normalize else np.abs(true_count - pred_count)


class PredictionErrorMetric(Metric):

    def __init__(self, num_classes: int, normalize: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.normalize = normalize
        self.total_cm = self.add_weight("total", shape=(num_classes, num_classes), initializer="zeros",
                                        dtype=tf.int32)

    def update_state(self, y_true: Iterable, y_pred: Iterable, *args, **kwargs):
        self.total_cm.assign_add(tf.math.confusion_matrix(tf.argmax(y_true, 1, output_type=tf.int32),
                                                          tf.argmax(y_pred, 1, output_type=tf.int32),
                                                          dtype=tf.int32, num_classes=self.num_classes))
        return self.total_cm

    def result(self):
        true_count = tf.reduce_sum(tf.reduce_sum(input_tensor=self.total_cm, axis=1) *
                                   tf.range(start=0, limit=self.num_classes, dtype=tf.int32))
        pred_count = tf.reduce_sum(tf.reduce_sum(input_tensor=self.total_cm, axis=0) *
                                   tf.range(start=0, limit=self.num_classes, dtype=tf.int32))
        return tf.cond(pred=tf.cast(self.normalize, dtype=tf.bool),
                       true_fn=lambda: tf.divide(tf.abs(true_count - pred_count), true_count),
                       false_fn=lambda: tf.cast(tf.abs(true_count - pred_count), dtype=tf.float64))

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape, dtype=tf.int32))


class Evaluation:
    cls_metrics = {'accuracy': metrics.accuracy_score, 'precision': metrics.precision_score,
                   'recall': metrics.recall_score, 'f1': metrics.f1_score, 'balanced': metrics.balanced_accuracy_score,
                   'kappa': metrics.cohen_kappa_score, 'auc': metrics.roc_auc_score,
                   'abs_error': lambda y_true, y_pred: prediction_error(y_true, y_pred, normalize=False),
                   'rel_error': lambda y_true, y_pred: prediction_error(y_true, y_pred, normalize=True)}
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
