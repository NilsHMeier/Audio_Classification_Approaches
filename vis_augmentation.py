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
    # Apply data augmentation with noise and plot the results
    wave_aug = WaveAugmentation(noise_scale=0.05, factor_scale=0.5)
    sample_noise = wave_aug.apply_noise_np(audio_signal=sample)
    Visualizer.plot_augmentation_results(samples={'Original Sample': sample, 'Augmented Sample': sample_noise},
                                         sampling_rate=22050, title='Waveform Augmentation with Noise')
    # Apply data augmentation with factor and plot the results
    sample_factor = wave_aug.apply_factor_np(audio_signal=sample)
    Visualizer.plot_augmentation_results(samples={'Original Sample': sample, 'Augmented Sample': sample_factor},
                                         sampling_rate=22050, title='Waveform Augmentation with Factor')
    # Apply data augmentation with factor & noise and plot the results
    sample_augmented = wave_aug.apply_both_np(audio_signal=sample)
    Visualizer.plot_augmentation_results(samples={'Original Sample': sample, 'Augmented Sample': sample_augmented},
                                         sampling_rate=22050, title='Waveform Augmentation with Factor and Noise')
    # Plot all samples for direkt comparison
    Visualizer.plot_augmentation_results(samples={'Original Sample': sample, 'Noise Augmentation': sample_noise,
                                                  'Factor Augmentation': sample_factor, 'Combined': sample_augmented},
                                         sampling_rate=22050, title='Comparison of Waveform Augmentation Techniques')


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
    Visualizer.plot_augmentation_results({'Original Sample': sample, 'Augmented Sample': sample_augmented},
                                         sampling_rate=22050, title='Spectrogram Augmentation')


if __name__ == '__main__':
    show_wave_augmentation()
    show_spectral_augmentation()
