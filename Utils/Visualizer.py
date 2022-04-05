import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import librosa
from librosa import display
import cv2
from typing import List, Tuple, Dict
from tensorflow.keras.callbacks import History


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
        axs[index].set(xlim=(np.min(dataset[x_column]), np.max(dataset[x_column])))
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


def plot_heatmap(matrix: np.ndarray, xlabel: str, xticks: List[str], ylabel: str, yticks: List[str], title: str = '',
                 suptitle: str = '', colormap: str = 'Blues'):
    colormap = plt.cm.get_cmap(colormap)
    fig, axs = plt.subplots()
    fig.suptitle(title, fontsize=16)
    sns.heatmap(matrix, xticklabels=xticks, yticklabels=yticks, annot=True, fmt='g', ax=axs, cmap=colormap)
    axs.set(xlabel=xlabel, ylabel=ylabel, title=suptitle)
    plt.show()


def plot_feature_performances(features: List[str], scores: Dict[str, List[float]], title: str = '', suptitle: str = ''):
    # Create a DataFrame out of the provided features and scores
    df = pd.DataFrame(columns=['Features', 'Score', 'Model'])
    for m, s in scores.items():
        df = df.append(other=pd.DataFrame({'Features': features, 'Score': s, 'Model': [m]*len(s)}), ignore_index=True)

    # Create subplot and add title
    fig, axs = plt.subplots()
    fig.suptitle(title, fontsize=16)

    # Create horizontal bar plot
    bars = sns.barplot(data=df, y='Features', x='Score', hue='Model', palette=sns.color_palette("Blues", as_cmap=False),
                       ax=axs)
    axs.set(title=suptitle, xlabel='Score', ylabel='Features')
    # Annotate the bars
    for bar in bars.containers:
        bars.bar_label(bar)
    plt.tight_layout()
    plt.show()


def plot_augmentation_results(samples: Dict[str, np.ndarray], sampling_rate: int,
                              title: str = ''):
    # Create figure and add subplots
    fig, axs = plt.subplots(nrows=len(samples), sharex=True)
    fig.suptitle(title, fontsize=16)

    # Iterate over dictionary and plot the sample
    for i, (name, sample) in enumerate(samples.items()):
        # Check type of input
        if len(sample.shape) == 1:
            # Waveform input
            librosa.display.waveplot(y=sample, sr=sampling_rate, ax=axs[i])
        elif len(sample.shape) == 2:
            # Spectrogram input
            sns.heatmap(sample, ax=axs[i], cmap='rocket')
        else:
            raise ValueError('Unknown input type! Input should be either 1-d for waveform or 2-d for spectrograms')

        # Set titles of subplots
        axs[i].set(title=name)
    plt.show()


def plot_trainings_history(history: History, params: List[str], single_plot: bool = False, validation: bool = False,
                           title: str = ''):
    # Create figure and subplots based on given parameters
    if single_plot or len(params) == 1:
        fig, axs = plt.subplots()
        axs = np.array([axs])
    else:
        fig, axs = plt.subplots(nrows=len(params))
    fig.suptitle(title, fontsize=16)

    # Create plots with params
    for index, param in enumerate(params):
        index = 0 if single_plot else index
        axs[index].plot(history.epoch, history.history[param], label=f'train_{param}')
        if validation:
            axs[index].plot(history.epoch, history.history[f'val_{param}'], label=f'val_{param}')
        if not single_plot:
            axs[index].legend()
        axs[index].set(xlabel='Epoch', ylabel=param.title(), xlim=(0, np.max(history.epoch)-1))
    if single_plot:
        axs[0].legend()
    plt.show()


def plot_learning_rate(history: History, plot_metric: bool = False, metric: str = ''):
    # Create figure and subplots
    if plot_metric:
        fig, axs = plt.subplots(nrows=2)
    else:
        fig, axs = plt.subplots()
        axs = np.array([axs])
    fig.suptitle('Learning Rate Optimization', fontsize=16)
    # Plot the results
    axs[0].semilogx(history.history['lr'], history.history['loss'])
    axs[0].set(title='Loss depending on Learning Rate', xlabel='Learning Rate', ylabel='Loss',
               xlim=(min(history.history['lr']), max(history.history['lr'])), ylim=(0, max(history.history['loss'])))
    if plot_metric:
        axs[1].semilogx(history.history['lr'], history.history[metric])
        axs[1].set(title='Accuracy depending on Learning Rate', xlabel='Learning Rate', ylabel='Accuracy',
                   xlim=(min(history.history['lr']), max(history.history['lr'])), ylim=(0, 1))
    plt.show()
