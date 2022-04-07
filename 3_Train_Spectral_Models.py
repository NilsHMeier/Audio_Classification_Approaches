import argparse
import itertools
import pathlib
import pandas as pd
import numpy as np
import tensorflow.keras as keras
from Preprocessing.FeatureEngineering import SpectralFeatures
from MachineLearning.ModelBuilder import SpectralModels as SM
from MachineLearning import Training
from MachineLearning.DataAugmentation import SpectrogramAugmentation
from Utils import Visualizer

# Set paths and sampling rate
AUDIO_PATH = pathlib.Path('data/Audio')
LABEL_PATH = pathlib.Path('data/Labels')
AUDIO_SR = 22050

# Set selected parameters for features
WINDOW_SIZE = 1.25
STEP_SIZE = 1.0
MODES = ['mels', 'stft']


def test_feature_sizes(binary_labels: bool):
    """
    Calculate the relative prediction error for different combinations of window size and step size and plot the results
    as a heatmap.

    :param binary_labels: Decide whether to use binary labels (0 or 1 for no car passed and car passed) or multiclass
    labels (one label for each passed number of cars).
    """
    # Create feature engineer with the given label mode
    feature_engineer = SpectralFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR,
                                        label_mode='max' if binary_labels else 'sum')
    augment_engineer = SpectrogramAugmentation(percentage=0.1)
    # Set parameters to test
    granularity = 0.25 if binary_labels else 2.5
    window_sizes = np.arange(0.25, 1.75, granularity) if binary_labels else np.arange(2.5, 12.5, granularity)
    min_step_size = np.min(window_sizes)
    # Create dataframe to save the results for later
    results = pd.DataFrame(index=window_sizes, columns=window_sizes)

    # Perform gridsearch over parameters
    count, overall = 1, sum(range(1, len(window_sizes) + 1))
    for window_size in window_sizes:
        for step_size in np.arange(min_step_size, window_size + granularity, granularity):
            print(f'Running training for window size {window_size} and step size {step_size} ({count}/{overall})')
            # Load features and labels
            features, labels = feature_engineer.features_from_directory(window_size=window_size, step_size=step_size)
            class_count = keras.utils.to_categorical(labels).shape[1]

            # Get model and get best score out of multiple runs
            model = SM.build_complex_cnn(input_shape=features[0].shape, num_classes=class_count, print_summary=False)
            score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error',
                                               aggregation_mode='min', augmentation_fn=augment_engineer.apply_tf)
            results.at[step_size, window_size] = score
            count += 1
    classnames = list(map(str, results.columns))
    Visualizer.plot_heatmap(matrix=results.to_numpy(dtype=np.float).round(decimals=4), xlabel='Window Size',
                            xticks=classnames, ylabel='Step Size', yticks=classnames, title='Feature Size Optimization',
                            suptitle='Relative Error of different combinations', colormap='Blues')
    if ARGS.save:
        results.to_csv(f'Results/Spec/Sizes_{"Binary" if binary_labels else "Multi"}.csv')


def test_feature_modes():
    """
    Calculate the relative prediction error for different feature mode combinations with all available models and
    plot the results as a grouped barchart.
    """
    # Create feature and augmentation engineer
    feature_engineer = SpectralFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR)
    augment_engineer = SpectrogramAugmentation(percentage=0.1)
    # Get available modes and build combinations
    modes = feature_engineer.available_modes
    combinations = []
    for i in np.arange(1, len(modes) + 1):
        combinations.extend(list(itertools.combinations(modes, i)))

    scores = {'Base CNN': [], 'Complex CNN': [], 'Recurrent CNN': [], 'Residual CNN': []}
    # Load all available feature modes to optimize performance
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                modes=modes)
    class_count = keras.utils.to_categorical(labels).shape[1]

    # Iterate over all combinations and save the score for plotting
    for i, feature_modes in enumerate(combinations):
        print(f'Running training for features {feature_modes} ({i+1}/{len(combinations)})')
        # Slice relevant features from available features
        selected_features = features[:, :, :, [modes.index(m) for m in feature_modes]]

        # Iterate over models with the same method interface
        for build_model, model_name in zip([SM.build_base_cnn, SM.build_complex_cnn, SM.build_recurrent_cnn_model],
                                           ['Base CNN', 'Complex CNN', 'Recurrent CNN']):
            print(f'Current model type is {model_name}')
            # Build the model
            model = build_model(input_shape=selected_features[0].shape, num_classes=class_count, print_summary=False)
            score = Training.get_stable_metric(model=model, features=selected_features, labels=labels,
                                               metric='rel_error', aggregation_mode='min',
                                               augmentation_fn=augment_engineer.apply_tf)
            scores[model_name].append(np.round(score, decimals=4))

        # Train the residual model with 3 residual blocks
        print('Current model type is Residual CNN')
        model = SM.build_adapted_residual_model(input_shape=selected_features[0].shape, residual_blocks=3,
                                                num_classes=class_count, print_summary=False)
        score = Training.get_stable_metric(model=model, features=selected_features, labels=labels, metric='rel_error',
                                           aggregation_mode='min', augmentation_fn=augment_engineer.apply_tf)
        scores['Residual CNN'].append(np.round(score, decimals=4))

    # Join combinations to get feature labels
    feature_names = ['-'.join(c) for c in combinations]
    Visualizer.plot_feature_performances(features=feature_names, scores=scores, title='Feature Mode Optimization',
                                         suptitle='Relative Error of different combinations')
    if ARGS.save:
        pd.DataFrame(data=scores, index=feature_names).to_csv('Results/Spec/Feature_Modes.csv')


def test_data_augmentation():
    """
    Calculate the relative prediction error for different augmentation percentages with all available models and plot
    the results as a linegraph.
    """
    # Create feature engineer and load the features
    feature_engineer = SpectralFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR,
                                        standardize=True)
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                modes=MODES)
    class_count = keras.utils.to_categorical(labels).shape[1]
    models = [SM.build_base_cnn(input_shape=features[0].shape, num_classes=class_count, print_summary=False),
              SM.build_complex_cnn(input_shape=features[0].shape, num_classes=class_count, print_summary=False),
              SM.build_recurrent_cnn_model(input_shape=features[0].shape, num_classes=class_count, print_summary=False),
              SM.build_adapted_residual_model(input_shape=features[0].shape, num_classes=class_count, residual_blocks=3,
                                              print_summary=False)]
    # Create DataFrame to save the results
    result_df = pd.DataFrame(index=np.linspace(0, 1, 11),
                             columns=['Base CNN', 'Complex CNN', 'Recurrent CNN', 'Residual CNN'])

    # Iterate over index of DataFrame to use the values as percentage for data augmentation
    for i, percentage in enumerate(result_df.index):
        print(f'Running training with {percentage} ({i + 1}/{len(result_df.index)})')
        augmentation = SpectrogramAugmentation(percentage=percentage)
        for model, model_name in zip(models, result_df.columns):
            score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error',
                                               aggregation_mode='min', augmentation_fn=augmentation.apply_tf)
            result_df.at[percentage, model_name] = score

    # Plot the results
    Visualizer.plot_data(dataset=result_df.reset_index(), plot_columns=[list(result_df.columns)], x_column='index',
                         types=['plot'], title='Data Augmentation Impact')
    if ARGS.save:
        result_df.to_csv('Results/Spec/data_aug.csv')


def test_resnet_architectures():
    """
    Calculate the relative prediction error for different number of residual blocks with  models available residual
    models and plot the results as a grouped barchart.
    """
    # Create feature and augmentation engineer
    feature_engineer = SpectralFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR)
    augment_engineer = SpectrogramAugmentation(percentage=0.1)
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                modes=MODES)
    class_count = keras.utils.to_categorical(labels).shape[1]
    # Get available modes and build combinations
    n_blocks = np.arange(1, 6, 1)

    scores = {'Residual CNN': [], 'Adapted Residual CNN': []}

    # Iterate over all n_blocks
    for i, n in enumerate(n_blocks):
        print(f'Running training with {n} residual blocks ({i + 1}/{len(n_blocks)})')

        # Train standard residual model
        model = SM.build_residual_model(input_shape=features[0].shape, num_classes=class_count, residual_blocks=n,
                                        print_summary=False)
        score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error',
                                           aggregation_mode='min', augmentation_fn=augment_engineer.apply_tf)
        scores['Residual CNN'].append(score)

        # Train adapted residual model
        model = SM.build_adapted_residual_model(input_shape=features[0].shape, num_classes=class_count,
                                                residual_blocks=n, print_summary=False)
        score = Training.get_stable_metric(model=model, features=features, labels=labels, metric='rel_error',
                                           aggregation_mode='min', augmentation_fn=augment_engineer.apply_tf)
        scores['Adapted Residual CNN'].append(score)

    # Join combinations to get feature labels
    block_str = [str(i) for i in n_blocks]
    Visualizer.plot_feature_performances(features=block_str, scores=scores, title='Residual Network Optimization',
                                         suptitle='Relative Error depending on Number of Residual Blocks')
    if ARGS.save:
        pd.DataFrame(data=scores, index=block_str).to_csv('Results/Spec/Residual_Models.csv')


def main():
    if ARGS.sizes:
        if ARGS.sizemode in ['binary', 'both']:
            test_feature_sizes(binary_labels=True)
        if ARGS.sizemode in ['multi', 'both']:
            test_feature_sizes(binary_labels=False)
    if ARGS.modes:
        test_feature_modes()
    if ARGS.augment:
        test_data_augmentation()
    if ARGS.resnet:
        test_resnet_architectures()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=bool, default=True, choices=[True, False],
                        help='Choose to test different window and step sizes.')
    parser.add_argument('--sizemode', type=str, default='both', choices=['binary', 'multi', 'both'],
                        help='Set which label mode(s) to use for feature size optimization.')
    parser.add_argument('--modes', type=bool, default=True, choices=[True, False],
                        help='Choose to test different feature modes.')
    parser.add_argument('--augment', type=bool, default=True, choices=[True, False],
                        help='Choose to study the impact of data augmentation.')
    parser.add_argument('--resnet', type=bool, default=True, choices=[True, False],
                        help='Choose to test different numbers of residual blocks.')
    parser.add_argument('--save', type=bool, default=True, choices=[True, False],
                        help='Select to save the results to external files.')
    ARGS, unparsed = parser.parse_known_args()
    # Print out args
    for key, value in vars(ARGS).items():
        print(f'{key} : {value}')
    main()
