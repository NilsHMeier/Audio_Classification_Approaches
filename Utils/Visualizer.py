import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
from librosa import display
import cv2
from typing import List, Tuple


def plot_data(dataset: pd.DataFrame, plot_columns: List[List[str]], x_column: str, types: List[str], title: str = '',
              shareX: bool = True, shareY: bool = False):
    fig, axs = plt.subplots(nrows=len(plot_columns), sharex=shareX, sharey=shareY)
    if len(plot_columns) == 1:
        axs = [axs]
    fig.suptitle(title, fontsize=16)
    for index, columns in enumerate(plot_columns):
        for col in columns:
            if types[index] == 'plot':
                axs[index].plot(dataset[x_column], dataset[col], label=col)
            elif types[index] == 'scatter':
                axs[index].scatter(dataset[x_column], dataset[col], label=col)
            else:
                print(f'Unknown plot type "{types[index]}"!')
        axs[index].legend()
        axs[index].set(xlim=(0, np.max(dataset[x_column])))
    plt.show()


def plot_clusters_3d(dataset: pd.DataFrame, plot_columns: List[Tuple[str, str, str]], cluster_col: str,
                     labels: List[str], colors: List[str], title: str = ''):
    # Create figure and add 3d subplots
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    subplots = []
    for i in range(len(plot_columns)):
        subplot = int(f'{1}{len(plot_columns)}{i + 1}')
        subplots.append(fig.add_subplot(subplot, projection='3d'))
    # Scatter points in each subplot using the set attributes
    for index, ax in enumerate(subplots):
        legend_dict = {}
        columns = plot_columns[index]
        ax.set(xlabel=columns[0], ylabel=columns[1], zlabel=columns[2])
        for i in dataset.index:
            cluster = dataset.loc[i, cluster_col]
            pt = ax.scatter(dataset.loc[i, columns[0]], dataset.loc[i, columns[1]], dataset.loc[i, columns[2]],
                            color=colors[cluster])
            if cluster not in legend_dict:
                legend_dict[cluster] = pt
        ax.legend(legend_dict.values(), labels)
    plt.show()


def plot_audio(audio: np.ndarray, sampling_rate: int, modes: List[str], plot_colorbar: bool = False, title: str = ''):
    supported_modes = ['waveplot', 'mfcc', 'melspectrogram', 'chroma_stft', 'chroma_cqt']
    modes = [mode for mode in modes if mode in supported_modes]
    fig, axs = plt.subplots(nrows=len(modes), sharex=True, sharey=False)
    fig.suptitle(title, fontsize=16)
    if len(modes) == 1:
        axs = [axs]
    for index, mode in enumerate(modes):
        img = None
        if mode == 'waveplot':
            librosa.display.waveplot(y=audio, sr=sampling_rate, ax=axs[index])
            axs[index].set(title='Waveplot')
        elif mode == 'mfcc':
            mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate)
            img = librosa.display.specshow(data=mfccs, x_axis='time', y_axis='log', ax=axs[index])
            axs[index].set(title='Mel-Frequency-Cepstral-Coefficients')
        elif mode == 'melspectrogram':
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sampling_rate, n_mels=64)
            spectrogram_db = librosa.power_to_db(S=spectrogram, ref=np.max)
            print(f'Melspectrogram shape = {spectrogram_db.shape}')
            img = librosa.display.specshow(data=spectrogram_db, x_axis='time', y_axis='mel', ax=axs[index])
            axs[index].set(title='Melspectrogram')
        elif mode == 'chroma_stft':
            chroma = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)
            print(f'Chroma-STFT shape = {chroma.shape}')
            img = librosa.display.specshow(data=chroma, x_axis='time', y_axis='chroma', ax=axs[index])
            axs[index].set(title='Chromagram-STFT')
        elif mode == 'chroma_cqt':
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sampling_rate)
            print(f'Chroma-CQT shape = {chroma.shape}')
            img = librosa.display.specshow(data=chroma, x_axis='time', y_axis='chroma', ax=axs[index])
            axs[index].set(title='Chroma-CQT')
        if plot_colorbar and img is not None:
            fig.colorbar(img, ax=axs[index], format='%+2.0f dB')
    plt.show()


def display_images(images: List[np.ndarray], titles: List[str], plot_shape: Tuple[int, int] = None, title: str = ''):
    if plot_shape is not None:
        fig, axs = plt.subplots(nrows=plot_shape[0], ncols=plot_shape[1], sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(nrows=(len(images)), sharex=True, sharey=True)
    axs = np.array(axs).flatten()
    fig.suptitle(title, fontsize=16)
    for index, image in enumerate(images):
        axs[index].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[index].set(title=titles[index])
    plt.tight_layout()
    plt.show()
