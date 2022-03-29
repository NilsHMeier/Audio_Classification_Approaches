import pathlib
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, patches
from Utils import Visualizer
from Preprocessing.AudioProcessor import AudioProcessor
from Preprocessing.FeatureEngineering import SpectralFeatures

# Set paths and create objects
AUDIO_PATH = pathlib.Path('data/Audio')
VIDEO_PATH = pathlib.Path('data/Video')
LABEL_PATH = pathlib.Path('data/Labels')
FILENAME = 'Sample_15'
SAMPLING_RATE = 22050
WINDOW_SIZE = 1.5
STEP_SIZE = 1.0
audio_engineer = AudioProcessor(video_path=VIDEO_PATH, audio_path=AUDIO_PATH, target_sr=SAMPLING_RATE)
feature_engineer = SpectralFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH, audio_sr=SAMPLING_RATE)


def visualize_spectrogram_features():
    audio, sr = audio_engineer.load_audio_from_wav(audio_name=FILENAME)
    Visualizer.plot_audio(audio=audio, sampling_rate=sr,
                          modes=['waveplot', 'melspectrogram', 'chroma_stft', 'chroma_cqt'],
                          title='Spectrogram-Based Features')


def visualize_handcrafted_features():
    # Load audio signal and label file
    audio, sr = audio_engineer.load_audio_from_wav(audio_name=FILENAME)
    label_df = feature_engineer.load_label_file(filename=FILENAME)

    # Create new dataframe for features
    feature_df = pd.DataFrame(columns=['max_freq', 'freq_weighted', 'pse', 'label'])
    snippet_length = int(WINDOW_SIZE * sr)
    frequencies = np.fft.rfftfreq(snippet_length, 1 / sr).round(3)
    fft_results = {1: [], 0: []}
    for index, timestamp in enumerate(np.arange(0, label_df['time'].max(), STEP_SIZE)):
        # Select relevant audio
        start_point = int(timestamp * sr)
        relevant_audio = audio[start_point:start_point + snippet_length]
        if len(relevant_audio) != snippet_length:
            break

        # Apply fourier transformation and calculate features
        amplitude = np.fft.rfft(relevant_audio, len(relevant_audio)).real
        max_freq = frequencies[np.argmax(np.abs(amplitude) * frequencies)]
        freq_weighted = float(np.sum(frequencies * amplitude)) / np.sum(amplitude)
        psd = np.divide(np.square(amplitude), float(len(amplitude)))
        psd_pdf = np.divide(psd, np.sum(psd))
        pse = -np.sum(np.log(psd_pdf) * psd_pdf) if np.count_nonzero(psd_pdf) == psd_pdf.size else 0

        # Get label
        label = label_df[np.logical_and(timestamp <= label_df['time'],
                                        label_df['time'] < timestamp + WINDOW_SIZE)]['label'].max()

        # Save the fft results to visualize them later
        fft_results[int(label)].append(amplitude)

        # Append features to feature dataframe
        feature_df = feature_df.append({'max_freq': max_freq, 'freq_weighted': freq_weighted, 'pse': pse,
                                        'label': label}, ignore_index=True)

    # Convert label column to integer and plot the results
    feature_df['label'] = feature_df['label'].astype(int)
    Visualizer.plot_clusters_3d(dataset=feature_df, plot_columns=[('max_freq', 'freq_weighted', 'pse')],
                                cluster_col='label', labels=['No car passed', 'Car passed'], colors=['b', 'r'],
                                title='Handcrafted Features')

    # Visualize two random results of fourier transformation
    fft_df = pd.DataFrame(data={'freqs': frequencies,
                                'Car passed': fft_results[1][random.randint(0, len(fft_results[1]))],
                                'No Car passed': fft_results[0][random.randint(0, len(fft_results[0]))]})
    Visualizer.plot_data(dataset=fft_df, plot_columns=[['Car passed', 'No Car passed']], x_column='freqs',
                         types=['plot'], title='Results of Fourier Transformation')


def visualize_feature_extraction():
    # Load audio signal and label file
    audio, sr = audio_engineer.load_audio_from_wav(audio_name='Sample_07')
    label_df = feature_engineer.load_label_file(filename='Sample_07')
    # Use first 25s of sample to demeonstrate the extraction
    audio = audio[:15 * sr]
    label_df = label_df[label_df['time'] <= 15]

    # Create figure and plot the waveform & the labels
    fig, axs = plt.subplots(figsize=(16, 7))
    fig.suptitle('Feature Extraction', fontsize=16)
    axs.plot(np.arange(len(audio)) / sr, audio, label='Audio Signal', color='lightblue', zorder=1)
    axs.vlines(label_df.loc[label_df['label'] == 1, 'time'] + 0.1, ymin=min(audio), ymax=max(audio), color='red',
               label='Vehicle Passes', linestyles='dotted', zorder=2)
    # Plot the boxes for the respective windows
    for i in range(15):
        point = (i * STEP_SIZE, -0.04 if i % 2 == 0 else -0.038)
        box = patches.Rectangle(xy=point, width=WINDOW_SIZE, height=0.08, color='darkblue', fill=False, zorder=3,
                                label='Windows & Label' if i == 0 else None)
        axs.add_patch(box)
        label = label_df[(label_df['time'] >= i * STEP_SIZE) & (label_df['time'] < i * STEP_SIZE + WINDOW_SIZE)][
            'label'].sum()
        axs.text(x=point[0] + 0.7, y=-0.044 if i % 2 == 0 else 0.044, s=str(label), zorder=4)

    # Format the plot
    axs.legend()
    axs.set(title=f'Window Size = {WINDOW_SIZE} - Step Size = {STEP_SIZE}', xlim=(0, 15), ylim=(-0.06, 0.06))
    plt.show()


if __name__ == '__main__':
    visualize_spectrogram_features()
    visualize_handcrafted_features()
    visualize_feature_extraction()
