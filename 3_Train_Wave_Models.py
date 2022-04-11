import argparse
import itertools
import pathlib
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from Preprocessing.FeatureEngineering import WaveFeatures
from MachineLearning import Training
from MachineLearning.Evaluation import Evaluation
from MachineLearning.ModelBuilder import WaveModels as WM
from MachineLearning.DataAugmentation import WaveAugmentation
from Utils import Visualizer

# Set paths and create feature engineer
AUDIO_PATH = pathlib.Path('data/Audio')
LABEL_PATH = pathlib.Path('data/Labels')
AUDIO_SR = 22050

# Set selected parameters for features
WINDOW_SIZE = 1.25
STEP_SIZE = 1.0


def test_wavenet_architecture():
    """
    Calculate the relative prediction error for different combinations of k_layers and n_filters and plot the results
    as a heatmap.
    """
    # Create feature & data augmentation instances and load the features
    feature_engineer = WaveFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR, label_mode='max')
    augment_engineer = WaveAugmentation(noise_scale=0.05, factor_scale=0.1)
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                stack_data=True)
    class_count = keras.utils.to_categorical(labels).shape[1]

    # Set number of layers to test
    num_layers = np.arange(2, 6, 1)
    num_filter = np.array([8, 16, 24, 32])
    combinations = list(itertools.product(num_layers, num_filter))

    # Create DataFrame to save the results
    results = pd.DataFrame(index=num_layers, columns=num_filter)

    # Iterate over combinations
    for i, (k_layer, n_filter) in enumerate(combinations):
        print(f'Running training with {k_layer} layers and {n_filter} filters ({i + 1}/{len(combinations)}).')
        model = WM.build_wavenet_model(input_shape=features[0].shape, num_classes=class_count, k_layers=k_layer,
                                       num_filters=n_filter, compile_model=True, print_summary=False)
        score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error',
                                           aggregation_mode='min', augmentation_fn=augment_engineer.apply_both_tf)
        results.at[k_layer, n_filter] = score
    # Plot the results
    Visualizer.plot_heatmap(matrix=results.to_numpy(dtype=np.float).round(decimals=4), xlabel='Number of Filters',
                            ylabel='Number of Layers', xticks=num_filter.astype(str), yticks=num_layers.astype(str),
                            title='Optimization of WaveNet Architecture', suptitle='Relative Error per Combination',
                            colormap='Blues')
    if ARGS.save:
        results.to_csv('Results/Wave/WaveNet.csv')


def test_parallel_cnn_architecture():
    """
    Calculate the relative predictione error for different combinations of kernel sizes and plot the results as a
    grouped bar chart.
    """
    # Create feature & data augmentation instances and load the features
    feature_engineer = WaveFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR, label_mode='max')
    augment_engineer = WaveAugmentation(noise_scale=0.1, factor_scale=0.15)
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                stack_data=True)
    class_count = keras.utils.to_categorical(labels).shape[1]

    # Set kernel sizes to test alone and in combination
    kernel_sizes = [4, 8, 16, 32]
    combinations = []
    for i in np.arange(1, len(kernel_sizes) + 1):
        combinations.extend(list(itertools.combinations(kernel_sizes, i)))
    feature_names = ['-'.join(list(map(str, c))) for c in combinations]
    print(feature_names)

    # Create dictionary to save the results
    scores = {'Addition': [], 'Concatenation': []}

    # Iterate over combinations
    for i, sizes in enumerate(combinations):
        print(f'Running training with kernel sizes {sizes} ({i + 1}/{len(combinations)}).')
        print('Current model type is Addition.')
        model = WM.build_parallel_cnn(input_shape=features[0].shape, num_classes=class_count, kernel_sizes=sizes,
                                      n_convs=5, aggregation_mode='add', print_summary=False)
        score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error', repeats=3,
                                           aggregation_mode='min', augmentation_fn=augment_engineer.apply_both_tf)
        scores['Addition'].append(score)

        print('Current model type is Concatenate.')
        model = WM.build_parallel_cnn(input_shape=features[0].shape, num_classes=class_count, kernel_sizes=sizes,
                                      n_convs=5, aggregation_mode='concat', print_summary=False)
        score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error', repeats=3,
                                           aggregation_mode='min', augmentation_fn=augment_engineer.apply_both_tf)
        scores['Concatenation'].append(score)

    # Join combinations to get feature labels
    feature_names = ['-'.join(list(map(str, c))) for c in combinations]
    Visualizer.plot_feature_performances(features=feature_names, scores=scores, title='Feature Mode Optimization',
                                         suptitle='Relative Error of different combinations')
    if ARGS.save:
        pd.DataFrame(data=scores, index=feature_names).to_csv('Results/Wave/Parallel_CNN.csv')


def test_data_augmentation():
    """
    Study the impact of data augmentation by training the WaveNet architecture with different combinations of factor and
    noise scale and visualize the results as a heatmap.
    """
    # Create feature engineer and load the features
    feature_engineer = WaveFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR, label_mode='max')
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                stack_data=True)
    labels_enc = keras.utils.to_categorical(labels)
    class_count = labels_enc.shape[1]

    # Set number of layers to test
    f_scales = np.array([0.0, 0.05, 0.1, 0.2])
    n_scales = np.array([0.0, 0.05, 0.1, 0.2])
    combinations = list(itertools.product(f_scales, n_scales))

    # Create DataFrame to save the results
    results = pd.DataFrame(index=f_scales, columns=n_scales)

    # Iterate over combinations
    for i, (f_scale, n_scale) in enumerate(combinations):
        print(f'Running training Factor scale of {f_scale} and Noise scale of {n_scale} ({i + 1}/{len(combinations)}).')
        # Create augmentation engineer with respective scales
        augment_engineer = WaveAugmentation(noise_scale=n_scale, factor_scale=f_scale)
        # Split the data and create TensorFlow dataset with augmentation
        X_train, X_test, y_train, y_test = train_test_split(features, labels_enc, test_size=0.3)
        dataset = Training.create_tf_dataset(X=X_train, y=y_train, augmentation_fn=augment_engineer.apply_both_tf)
        # Create the model and fit it using EarlyStopping, but do not restore the best weights
        model = WM.build_wavenet_model(input_shape=features[0].shape, num_classes=class_count, k_layers=4,
                                       num_filters=16, compile_model=True, print_summary=False)
        model.fit(dataset, validation_data=(X_test, y_test), epochs=100,
                  callbacks=Training.create_callbacks(patience=5, restore_weights=False, use_clr=True))
        score = Evaluation.single_classification(stat='rel_error', y_true=np.argmax(y_test, axis=1),
                                                 y_pred=np.argmax(model.predict(X_test), axis=1))
        results.at[f_scale, n_scale] = score
    # Plot the results
    Visualizer.plot_heatmap(matrix=results.to_numpy(dtype=np.float).round(decimals=4), xlabel='Noise Scale',
                            ylabel='Factor Scale', xticks=n_scales.astype(str), yticks=f_scales.astype(str),
                            title='Data Augmentation Impact', suptitle='Relative Error per Combination',
                            colormap='Blues')
    if ARGS.save:
        results.to_csv('Results/Wave/Data_Augmentation.csv')


def main():
    if ARGS.wavenet:
        test_wavenet_architecture()
    if ARGS.parallel:
        test_parallel_cnn_architecture()
    if ARGS.augment:
        test_data_augmentation()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavenet', type=bool, default=True, choices=[True, False],
                        help='Choose to test different WaveNet architectures.')
    parser.add_argument('--parallel', type=bool, default=True, choices=[True, False],
                        help='Choose to test different kernel sizes.')
    parser.add_argument('--augment', type=bool, default=True, choices=[True, False],
                        help='Choose to study the impact of data augmentation.')
    parser.add_argument('--save', type=bool, default=True, choices=[True, False],
                        help='Select to save the results to external files.')
    ARGS, unparsed = parser.parse_known_args()
    # Print out args
    for key, value in vars(ARGS).items():
        print(f'{key} : {value}')
    main()
