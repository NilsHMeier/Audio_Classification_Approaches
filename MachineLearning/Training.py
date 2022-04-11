from .Evaluation import Evaluation
from .ModelBuilder import compile_model_default
from .CLR import CyclicLR
import tensorflow as tf
from tensorflow.keras import models, callbacks, Model, utils
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from typing import Iterable, Tuple, Union, List, Callable, Dict

# Set overall parameters
BATCH_SIZE = 64
EPOCHS = 100
TEST_SIZE = 0.3


def get_best_model(model: Model, features: np.ndarray, labels: np.ndarray, scoring_metric: str, metrics: List[str],
                   mode: str = 'auto', repeats: int = 3, augmentation_fn: Callable = None, use_clr: bool = True) \
        -> Tuple[Model, float, callbacks.History, Dict[str, float]]:
    # Encode labels if required
    if len(labels.shape) != 2:
        labels = utils.to_categorical(labels)
    # Determine mode if required
    if mode == 'auto':
        mode = 'min' if scoring_metric in list(Evaluation.reg_metrics.keys()) + ['abs_error', 'rel_error'] else 'max'

    best_score, best_model, history, metric = 0.0 if mode == 'max' else np.inf, None, None, None
    # Train multiple models and return the model with the best performance
    for i in range(repeats):
        print(f'Performing run {i + 1}/{repeats}')
        # Split data and create TensorFlow dataset
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)
        dataset = create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augmentation_fn, batch_size=BATCH_SIZE)
        # Fit the model
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_history = temp_model.fit(dataset, epochs=EPOCHS, validation_data=(X_test, y_test),
                                      callbacks=create_callbacks(use_clr=use_clr))
        # Predict the test set and calculate the metric
        y_pred = np.argmax(temp_model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        score = Evaluation.single_classification(stat=scoring_metric, y_true=y_true, y_pred=y_pred)
        if mode == 'max' and score > best_score or mode == 'min' and score < best_score:
            best_score = score
            best_model = temp_model
            history = temp_history
            metric = Evaluation.multi_classification(stats=metrics, y_true=y_true, y_pred=y_pred)
    return best_model, best_score, history, metric


def get_stable_metric(model: Model, features: np.ndarray, labels: np.ndarray, metric: str, repeats: int = 3,
                      aggregation_mode: str = 'mean', augmentation_fn: Callable = None, use_clr: bool = True) -> float:
    # Encode labels if required
    if len(labels.shape) != 2:
        labels = utils.to_categorical(labels)
    scores = []
    for i in range(repeats):
        print(f'Performing run {i + 1}/{repeats}')
        # Split data and create TensorFlow dataset
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=TEST_SIZE)
        dataset = create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augmentation_fn, batch_size=BATCH_SIZE)
        # Fit the model
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_model.fit(dataset, epochs=EPOCHS, validation_data=(X_test, y_test),
                       callbacks=create_callbacks(use_clr=use_clr))
        # Predict the test set and calculate the metric
        y_pred = np.argmax(temp_model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        scores.append(Evaluation.single_classification(stat=metric, y_true=y_true, y_pred=y_pred))
    # Aggregate the scores using the given mode
    if aggregation_mode == 'mean':
        return np.mean(scores)
    elif aggregation_mode == 'median':
        return np.median(scores)
    elif aggregation_mode == 'max':
        return np.max(scores)
    elif aggregation_mode == 'min':
        return np.min(scores)
    else:
        raise ValueError(f'Unknown aggregation mode {aggregation_mode}!')


def get_cross_validation_score(model: Model, features: np.ndarray, labels: np.ndarray, metric: Union[str, List[str]],
                               augmentation_fn: Callable = None, n_folds: int = 5, use_clr: bool = True)\
        -> Union[float, Dict[str, float]]:
    # Encode labels if required
    if len(labels.shape) != 2:
        labels = utils.to_categorical(labels)

    scores = [] if isinstance(metric, str) else {m: [] for m in metric}
    fold = 1
    for train, test in KFold(n_splits=n_folds, shuffle=True).split(X=features, y=labels):
        print(f'Running fold {fold} out of {n_folds}...')
        fold += 1
        # Extract train and test data and create TensorFlow dataset
        X_train, X_test, y_train, y_test = features[train], features[test], labels[train], labels[test]
        dataset = create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augmentation_fn, batch_size=BATCH_SIZE)
        # Fit the model
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_model.fit(dataset, validation_data=(X_test, y_test), epochs=EPOCHS,
                       callbacks=create_callbacks(use_clr=use_clr))
        # Predict the test set and calculate metrics
        y_pred = np.argmax(temp_model.predict(X_test), axis=1)
        y_true = np.argmax(y_test, axis=1)
        if isinstance(metric, str):
            scores.append(Evaluation.single_classification(stat=metric, y_true=y_true, y_pred=y_pred))
        else:
            for m, v in Evaluation.multi_classification(stats=metric, y_true=y_true, y_pred=y_pred).items():
                scores[m].append(v)
    return np.mean(scores) if isinstance(metric, str) else {m: np.mean(s) for m, s in scores.items()}


def find_optimal_learning_rate(model: Model, features: Union[np.ndarray, Iterable], labels: Union[np.ndarray, Iterable],
                               minimum_lr: float = 1e-8, epochs: int = 70) -> callbacks.History:
    # Encode labels if not already done
    if len(labels.shape) != 2:
        labels = utils.to_categorical(labels)
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=minimum_lr), loss=CategoricalCrossentropy(from_logits=True),
                  metrics=[CategoricalAccuracy(name='accuracy')])
    # Create learning rate scheduler and fit the model
    lr_scheduler = callbacks.LearningRateScheduler(schedule=lambda epoch: minimum_lr * 10 ** (epoch / 10))
    history = model.fit(x=features, y=labels, epochs=epochs, batch_size=64, callbacks=[lr_scheduler])
    return history


def create_tf_dataset(X: np.ndarray, y: np.ndarray, augmentation_fn: Callable = None,
                      batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    if augmentation_fn is not None:
        X_dataset = tf.data.Dataset.from_tensor_slices(X).map(map_func=augmentation_fn)
    else:
        X_dataset = tf.data.Dataset.from_tensor_slices(X)
    y_dataset = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((X_dataset, y_dataset)).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_callbacks(monitor: str = 'val_loss', mode: str = 'auto', patience: int = 10, use_clr: bool = True,
                     restore_weights: bool = True) -> List[callbacks.Callback]:
    early_stopping = callbacks.EarlyStopping(monitor=monitor, patience=patience, mode=mode,
                                             restore_best_weights=restore_weights)
    clr = CyclicLR(base_lr=0.001, max_lr=0.003, mode='triangular', step_size=500)
    return [early_stopping, clr] if use_clr else [early_stopping]
