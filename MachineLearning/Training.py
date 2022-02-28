from .Evaluation import Evaluation
from .ModelBuilder import compile_model_default
from tensorflow.keras import models, Sequential, callbacks, Model, utils
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import Iterable, Tuple, Union


def get_best_model(model: Model, X_train: Iterable, y_train: Iterable, X_test: Iterable,
                   y_test: Iterable, metric: str, repeats: int = 3) -> Tuple[Sequential, float, callbacks.History]:
    best_score, best_model, history = 0., None, None
    for i in range(repeats):
        print(f'Performing run {i + 1}/{repeats}')
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_history = temp_model.fit(x=X_train, y=y_train, batch_size=64, epochs=100, validation_split=0.3,
                                      callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        y_pred = np.argmax(model.predict(X_test), axis=1)
        score = Evaluation.single_classification(stat=metric, y_true=y_test, y_pred=y_pred)
        if score > best_score:
            best_score = score
            best_model = temp_model
            history = temp_history
    return best_model, best_score, history


def get_stable_metric(model: Model, X_train: Iterable, y_train: Iterable, X_test: Iterable,
                      y_test: Iterable, metric: str, repeats: int = 3, mode: str = 'mean') -> float:
    scores = []
    for i in range(repeats):
        print(f'Performing run {i + 1}/{repeats}')
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_model.fit(x=X_train, y=y_train, batch_size=64, epochs=50, validation_split=0.3,
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
                               metric: str, n_folds: int = 5) -> float:
    scores = []
    for train, test in StratifiedKFold(n_splits=n_folds, shuffle=True).split(X=features, y=labels):
        temp_model = compile_model_default(model=models.clone_model(model=model))
        temp_model.fit(x=features[train], y=labels[train], validation_split=0.3, epochs=100, batch_size=64,
                       callbacks=callbacks.EarlyStopping(patience=10, restore_best_weights=True))
        y_pred = np.argmax(model.predict(features[test]), axis=1)
        scores.append(Evaluation.single_classification(stat=metric, y_true=labels[test], y_pred=y_pred))
    return np.mean(scores)


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
