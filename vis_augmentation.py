from pathlib import Path
import numpy as np
from Preprocessing.FeatureEngineering import WaveFeatures, SpectralFeatures
from MachineLearning.DataAugmentation import WaveAugmentation, SpectrogramAugmentation
from Utils import Visualizer

# Set paths and select file
AUDIO_PATH = Path('data/Audio')
LABEL_PATH = Path('data/Labels')
FILENAME = 'Sample_15'

# Create preprocessing objects
wave_features = WaveFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH)
spec_features = SpectralFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH)


def show_wave_augmentation():
    # Load features and labels
    features, labels = wave_features.features_for_file(filename=FILENAME, window_size=0.5, step_size=0.5)
    # Select random sample where a car has passed
    sample = features[np.random.choice(np.where(labels == 1)[0])]
    # Apply data augmentation
    wave_aug = WaveAugmentation(scale=0.05)
    sample_augmented = wave_aug.apply_np(audio_signal=sample)
    # Plot the results
    Visualizer.plot_augmentation_results(original_sample=sample, augmented_sample=sample_augmented, sampling_rate=22050,
                                         title='Waveform Augmentation')


def show_spectral_augmentation():
    # Load features and labels
    features, labels = spec_features.features_for_file(filename=FILENAME, window_size=1.5, step_size=1.0,
                                                       modes=['stft'])
    # Select random sample where a car has passed
    sample = features[np.random.choice(np.where(labels == 1)[0])].squeeze()
    # Apply data augmentation
    spec_aug = SpectrogramAugmentation(percentage=0.2)
    sample_augmented = spec_aug.apply_np(spectrogram=sample)
    # Plot the results
    Visualizer.plot_augmentation_results(original_sample=sample, augmented_sample=sample_augmented, sampling_rate=22050,
                                         title='Spectrogram Augmentation')


if __name__ == '__main__':
    show_wave_augmentation()
    show_spectral_augmentation()
