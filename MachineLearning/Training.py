from .Evaluation import Evaluation
from .ModelBuilder import compile_model_default
import tensorflow as tf
from tensorflow.keras import models, Sequential, callbacks, Model, utils
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Iterable, Tuple, Union, List, Callable, Dict


def get_best_model(model: Model, X_train: Iterable, y_train: Iterable, X_test: Iterable, y_test: Iterable, metric: str,
                   repeats: int = 3, augmentation_fn: Callable = None) -> Tuple[Sequential, float, callbacks.History]:
    best_score, best_model, history = 0., None, None
    dataset = create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augmentation_fn)
    for i in range(repeats):
        print(f'Performing run {i + 1}/{repeats}')
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_history = temp_model.fit(dataset, batch_size=64, epochs=100, validation_data=(X_test, y_test),
                                      callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        score = Evaluation.single_classification(stat=metric, y_true=y_test, y_pred=y_pred)
        if score > best_score:
            best_score = score
            best_model = temp_model
            history = temp_history
    return best_model, best_score, history


def get_stable_metric(model: Model, X_train: Iterable, y_train: Iterable, X_test: Iterable, y_test: Iterable,
                      metric: str, repeats: int = 3, mode: str = 'mean', augmentation_fn: Callable = None) -> float:
    scores = []
    dataset = create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augmentation_fn)
    for i in range(repeats):
        print(f'Performing run {i + 1}/{repeats}')
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_model.fit(dataset, batch_size=64, epochs=100, validation_data=(X_test, y_test),
                       callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        scores.append(Evaluation.single_classification(stat=metric, y_true=y_test, y_pred=y_pred))
    if mode == 'mean':
        return np.mean(scores)
    elif mode == 'median':
        return np.median(scores)
    elif mode == 'max':
        return np.max(scores)
    elif mode == 'min':
        return np.min(scores)
    else:
        raise ValueError(f'Unknown mode {mode}!')


def get_cross_validation_score(model: Model, features: Union[np.ndarray, Iterable], labels: Union[np.ndarray, Iterable],
                               metric: Union[str, List[str]], augmentation_fn: Callable = None, n_folds: int = 5) \
        -> Union[float, Dict[str, float]]:
    if len(labels.shape) != 2:
        labels = utils.to_categorical(labels)

    scores = [] if isinstance(metric, str) else {m: [] for m in metric}
    fold = 1
    for train, test in StratifiedKFold(n_splits=n_folds, shuffle=True).split(X=features, y=np.argmax(labels, axis=1)):
        print(f'Running fold {fold} out of {n_folds}...')
        fold += 1
        X_train, X_test, y_train, y_test = features[train], features[test], labels[train], labels[test]
        temp_model = compile_model_default(model=models.clone_model(model=model))
        dataset = create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augmentation_fn)
        temp_model.fit(dataset, validation_data=(X_test, y_test), epochs=100,
                       callbacks=callbacks.EarlyStopping(patience=10, restore_best_weights=True))
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
                      batch_size: int = 64) -> tf.data.Dataset:
    if augmentation_fn is not None:
        X_dataset = tf.data.Dataset.from_tensor_slices(X).map(map_func=augmentation_fn)
    else:
        X_dataset = tf.data.Dataset.from_tensor_slices(X)
    y_dataset = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((X_dataset, y_dataset)).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
