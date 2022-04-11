import argparse
from pathlib import Path
import pandas as pd
import tensorflow.keras as keras
from Preprocessing.FeatureEngineering import SpectralFeatures as SF, WaveFeatures as WF
from MachineLearning.ModelBuilder import WaveModels as WM, SpectralModels as SM
from MachineLearning.DataAugmentation import WaveAugmentation as WA, SpectrogramAugmentation as SA
from MachineLearning import Training

# Set paths and constants
AUDIO_PATH = Path('data/Audio')
LABEL_PATH = Path('data/Labels')
SPEC_MODEL_PATH = Path('models/Spec')
WAVE_MODEL_PATH = Path('models/Wave')
AUDIO_SR = 22050
WINDOW_SIZE = 1.25
STEP_SIZE = 1.0


def train_spectral_models():
    # Create feature and augmentation instances
    feature_engineer = SF(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR, label_mode='max')
    augment_engineer = SA(percentage=0.2)

    # Load features with melspectrogram and chroma stft
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                modes=['mels', 'stft'])
    class_count = keras.utils.to_categorical(labels).shape[1]

    # Create DataFrame to save the results
    results = pd.DataFrame(index=['Base CNN', 'Complex CNN', 'Recurrent CNN', 'Adapted Residual CNN', 'Residual CNN'],
                           columns=['f1', 'recall', 'rel_error'])

    # Base CNN
    print('Current model is Base CNN (1/5)')
    sel_feat = features[:, :, :, [0]]
    model = SM.build_base_cnn(input_shape=sel_feat[0].shape, num_classes=class_count, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=sel_feat, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_tf)
    results.loc['Base CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=SPEC_MODEL_PATH.joinpath('Base_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=SPEC_MODEL_PATH.joinpath('Base_CNN.h5').__str__())

    # Complex CNN
    print('Current model is Complex CNN (2/5)')
    model = SM.build_complex_cnn(input_shape=features[0].shape, num_classes=class_count, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_tf)
    results.loc['Complex CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=SPEC_MODEL_PATH.joinpath('Complex_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=SPEC_MODEL_PATH.joinpath('Complex_CNN.h5').__str__())

    # Recurrent CNN
    print('Current model is Recurrent CNN (3/5)')
    sel_feat = features[:, :, :, [1]]
    model = SM.build_base_cnn(input_shape=sel_feat[0].shape, num_classes=class_count, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=sel_feat, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_tf)
    results.loc['Recurrent CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=SPEC_MODEL_PATH.joinpath('Recurrent_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=SPEC_MODEL_PATH.joinpath('Recurrent_CNN.h5').__str__())

    # Adapted Residual CNN
    print('Current model is Adapted Residual CNN (4/5)')
    model = SM.build_adapted_residual_model(input_shape=features[0].shape, residual_blocks=3, num_classes=class_count,
                                            print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_tf)
    results.loc['Adapted Residual CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=SPEC_MODEL_PATH.joinpath('Adapted_Residual_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=SPEC_MODEL_PATH.joinpath('Adapted_Residual_CNN.h5').__str__())

    # Residual CNN
    print('Current model is Residual CNN (4/5)')
    model = SM.build_residual_model(input_shape=features[0].shape, residual_blocks=2, num_classes=class_count,
                                    print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_tf)
    results.loc['Residual CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=SPEC_MODEL_PATH.joinpath('Residual_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=SPEC_MODEL_PATH.joinpath('Residual_CNN.h5').__str__())

    results.to_csv('Results/Spec/Model_Comparison.csv')


def train_wave_models():
    # Create feature and augmentation instances
    feature_engineer = WF(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=AUDIO_SR, label_mode='max')
    augment_engineer = WA(factor_scale=0.1, noise_scale=0.05)

    # Load features with melspectrogram and chroma stft
    features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=STEP_SIZE,
                                                                stack_data=True)
    class_count = keras.utils.to_categorical(labels).shape[1]

    # Create DataFrame to save the results
    results = pd.DataFrame(index=['Base CNN', 'Sample Level CNN', 'Recurrent CNN', 'Parallel CNN', 'WaveNet'],
                           columns=['f1', 'recall', 'rel_error'])

    # Base CNN
    print('Current model is Base CNN (1/5)')
    model = WM.build_base_model(input_shape=features[0].shape, num_classes=class_count, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_factor_tf)
    results.loc['Base CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=WAVE_MODEL_PATH.joinpath('Base_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=WAVE_MODEL_PATH.joinpath('Base_CNN.h5').__str__())

    # Sample Level CNN
    print('Current model is Sample Level CNN (2/5)')
    model = WM.build_sample_level_cnn(input_shape=features[0].shape, num_classes=class_count, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_factor_tf)
    results.loc['Sample Level CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=WAVE_MODEL_PATH.joinpath('Sample_Level_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=WAVE_MODEL_PATH.joinpath('Sample_Level_CNN.h5').__str__())

    # Recurrent CNN
    print('Current model is Recurrent CNN (3/5)')
    model = WM.build_recurrent_cnn_model(input_shape=features[0].shape, num_classes=class_count, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_factor_tf)
    results.loc['Recurrent CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=WAVE_MODEL_PATH.joinpath('Recurrent_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=WAVE_MODEL_PATH.joinpath('Recurrent_CNN.h5').__str__())

    # Parallel CNN
    print('Current model is Parallel CNN (4/5)')
    model = WM.build_parallel_cnn(input_shape=features[0].shape, num_classes=class_count, kernel_sizes=[4, 8, 32],
                                  aggregation_mode='concat', n_convs=5, print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_factor_tf)
    results.loc['Parallel CNN'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=WAVE_MODEL_PATH.joinpath('Parallel_CNN.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=WAVE_MODEL_PATH.joinpath('Parallel_CNN.h5').__str__())

    # WaveNet
    print('Current model is WaveNet (5/5)')
    model = WM.build_wavenet_model(input_shape=features[0].shape, num_classes=class_count, k_layers=4, num_filters=32,
                                   print_summary=False)
    model, score, history, metrics = Training.get_best_model(model=model, features=features, labels=labels,
                                                             scoring_metric='rel_error', repeats=5,
                                                             metrics=['f1', 'recall', 'rel_error'],
                                                             augmentation_fn=augment_engineer.apply_factor_tf)
    results.loc['WaveNet'] = metrics
    if ARGS.image:
        keras.utils.plot_model(model=model, to_file=WAVE_MODEL_PATH.joinpath('WaveNet.png').__str__())
    if ARGS.save:
        keras.models.save_model(model=model, filepath=WAVE_MODEL_PATH.joinpath('WaveNet.h5').__str__())

    results.to_csv('Results/Wave/Model_Comparison.csv')


def main():
    if ARGS.spectral:
        train_spectral_models()
    if ARGS.waveform:
        train_wave_models()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral', type=bool, default=False, choices=[True, False],
                        help='Choose to train spectral model architectures.')
    parser.add_argument('--waveform', type=bool, default=True, choices=[True, False],
                        help='Choose to train waveform model architectures.')
    parser.add_argument('--image', type=bool, default=True, choices=[True, False],
                        help='Select to save the model architectures as images.')
    parser.add_argument('--save', type=bool, default=True, choices=[True, False],
                        help='Select to save the models as .h5 files.')
    ARGS, unparsed = parser.parse_known_args()
    # Print out args
    for key, value in vars(ARGS).items():
        print(f'{key} : {value}')
    main()
