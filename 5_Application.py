import argparse
import pathlib
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as lm
from Preprocessing.FeatureEngineering import WaveFeatures, SpectralFeatures

# Set paths and constants
AUDIO_PATH = pathlib.Path('data/Audio')
LABEL_PATH = pathlib.Path('data/Test')
MODEL_PATH = pathlib.Path('models')
WINDOW_SIZE = 1.25
AUDIO_SR = 22050


def spectral_models():
    model_path = MODEL_PATH.joinpath('Spec')
    spec_models = {'Base CNN': lm(model_path.joinpath('Base_CNN.h5').__str__(), compile=False),
                   'Complex CNN': lm(model_path.joinpath('Complex_CNN.h5').__str__(), compile=False),
                   'Recurrent CNN': lm(model_path.joinpath('Recurrent_CNN.h5').__str__(), compile=False),
                   'Adapted Residual CNN': lm(model_path.joinpath('Adapted_Residual_CNN.h5').__str__(), compile=False),
                   'Residual CNN': lm(model_path.joinpath('Residual_CNN.h5').__str__(), compile=False)}

    results = pd.DataFrame(index=['Base CNN', 'Complex CNN', 'Recurrent CNN', 'Adapted Residual CNN', 'Residual CNN'],
                           columns=['Low', 'Low Error', 'Moderate', 'Moderate Error', 'High', 'High Error'])

    for density in os.listdir(LABEL_PATH):
        label_path = LABEL_PATH.joinpath(density)
        feature_engineer = SpectralFeatures(audio_path=AUDIO_PATH, label_path=label_path, audio_sr=AUDIO_SR)
        features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=WINDOW_SIZE,
                                                                    modes=['mels', 'stft'])
        real_vehicle_count = sum([pd.read_csv(label_path.joinpath(file))['label'].sum()
                                  for file in os.listdir(label_path)])
        print(f'Density {density} has {real_vehicle_count} vehicles (Binary Labels cover {sum(labels)})')

        for model_name, model in spec_models.items():
            if model_name == 'Base CNN':
                sel_features = features[:, :, :, [0]]
            elif model_name == 'Recurrent CNN':
                sel_features = features[:, :, :, [1]]
            else:
                sel_features = features
            y_pred = np.argmax(model.predict(sel_features), axis=1)
            abs_prediction, rel_error = sum(y_pred), abs(sum(y_pred)-real_vehicle_count) / real_vehicle_count
            results.loc[model_name, [f'{density}', f'{density} Error']] = [abs_prediction, rel_error]

    print(results)
    if ARGS.save:
        results.to_csv('Results/Spec/Testset_Prediction.csv')


def waveform_models():
    model_path = MODEL_PATH.joinpath('Wave')
    wave_models = {'Base CNN': lm(model_path.joinpath('Base_CNN.h5').__str__(), compile=False),
                   'Parallel CNN': lm(model_path.joinpath('Parallel_CNN.h5').__str__(), compile=False),
                   'Recurrent CNN': lm(model_path.joinpath('Recurrent_CNN.h5').__str__(), compile=False),
                   'Sample Level CNN': lm(model_path.joinpath('Sample_Level_CNN.h5').__str__(), compile=False),
                   'WaveNet': lm(model_path.joinpath('WaveNet.h5').__str__(), compile=False)}

    results = pd.DataFrame(index=['Base CNN', 'Parallel CNN', 'Recurrent CNN', 'Sample Level CNN', 'WaveNet'],
                           columns=['Low', 'Low Error', 'Moderate', 'Moderate Error', 'High', 'High Error'])

    for density in os.listdir(LABEL_PATH):
        label_path = LABEL_PATH.joinpath(density)
        feature_engineer = WaveFeatures(audio_path=AUDIO_PATH, label_path=label_path, audio_sr=AUDIO_SR)
        features, labels = feature_engineer.features_from_directory(window_size=WINDOW_SIZE, step_size=WINDOW_SIZE,
                                                                    stack_data=True)
        real_vehicle_count = sum([pd.read_csv(label_path.joinpath(file))['label'].sum()
                                  for file in os.listdir(label_path)])
        print(f'Density {density} has {real_vehicle_count} vehicles (Binary Labels cover {sum(labels)})')

        for model_name, model in wave_models.items():
            y_pred = np.argmax(model.predict(features), axis=1)
            abs_prediction, rel_error = sum(y_pred), abs(sum(y_pred) - real_vehicle_count) / real_vehicle_count
            results.loc[model_name, [f'{density}', f'{density} Error']] = [abs_prediction, rel_error]

    print(results)
    if ARGS.save:
        results.to_csv('Results/Wave/Testset_Prediction.csv')


def main():
    if ARGS.spectral:
        spectral_models()
    if ARGS.waveform:
        waveform_models()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spectral', type=bool, default=True, choices=[True, False],
                        help='Choose to predict test data with Spectral Models.')
    parser.add_argument('--waveform', type=bool, default=True, choices=[True, False],
                        help='Choose to predict test data with Waveform Models.')
    parser.add_argument('--save', type=bool, default=True, choices=[True, False],
                        help='Select to save the results as CSV files.')
    ARGS, unparsed = parser.parse_known_args()
    # Print out args
    for key, value in vars(ARGS).items():
        print(f'{key} : {value}')
    main()
