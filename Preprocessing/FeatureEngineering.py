import os
import io
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa
import cv2
from tqdm import tqdm
from Preprocessing.AudioProcessor import AudioProcessor
from typing import List, Tuple
from abc import abstractmethod, ABC


class FeatureEngineering(ABC):

    def __init__(self, audio_path: pathlib.Path, label_path: pathlib.Path, audio_sr: int, label_mode: str,
                 standardize: bool = False):
        self.audio_path = audio_path
        self.label_path = label_path
        self.label_mode = label_mode
        self.standardize = standardize
        self.audio_engineer = AudioProcessor(video_path=None, audio_path=audio_path, target_sr=audio_sr)

    def features_from_directory(self, window_size: float, step_size: float, modes: List[str] = None,
                                stack_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        all_features, all_labels = [], []
        print(f'Calculating features for files in directory {self.label_path}.')
        filenames = [file.split('.')[0] for file in os.listdir(self.label_path)]
        for file in tqdm(filenames):
            features, labels = self.features_for_file(filename=file, window_size=window_size, step_size=step_size,
                                                      modes=modes, stack_data=stack_data)
            all_features.extend(features)
            all_labels.extend(labels)
        return np.array(all_features), np.array(all_labels)

    def features_for_file(self, filename: str, window_size: float, step_size: float,
                          modes: List[str] = None, stack_data: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        # Load the audio file
        audio, sr = self.audio_engineer.load_audio_from_wav(audio_name=filename)

        # Use the class methods to get features and labels
        features = self.features_for_audio(audio=audio, sr=sr, window_size=window_size, step_size=step_size,
                                           modes=modes, stack_data=stack_data)
        labels = self.labels_for_file(filename=filename, window_size=window_size, step_size=step_size)

        # Make sure arrays have the same length
        if len(features) > len(labels):
            features = features[:len(labels)]
        elif len(features) < len(labels):
            labels = labels[:len(features)]

        return features, labels

    @abstractmethod
    def features_for_audio(self, audio: np.ndarray, sr: int, window_size: float, step_size: float,
                           modes: List[str] = None, stack_data: bool = False) -> np.ndarray:
        pass

    def labels_for_file(self, filename: str, window_size: float, step_size: float) -> np.ndarray:
        path = os.path.join(self.label_path, f'{filename}.csv')
        label_df = pd.read_csv(path, index_col=0)
        labels = []
        for timestamp in np.arange(0.0, label_df['time'].max(), step_size):
            # Check if car has passed in time window and append label
            relevant_rows = label_df[np.logical_and(timestamp <= label_df['time'],
                                                    label_df['time'] < timestamp + window_size)]['label']
            if self.label_mode == 'max':
                labels.append(relevant_rows.max())
            elif self.label_mode == 'sum':
                labels.append(relevant_rows.sum())
        return np.array(labels)

    def load_label_file(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.label_path, f'{filename}.csv')
        df = pd.read_csv(path, index_col=0)
        return df


class SpectralFeatures(FeatureEngineering):
    available_modes = ['mels', 'stft', 'cqt']

    def __init__(self, audio_path: pathlib.Path, label_path: pathlib.Path, audio_sr: int = 22050,
                 label_mode: str = 'max', standardize: bool = False):
        super().__init__(audio_path, label_path, audio_sr, label_mode, standardize)

    def features_for_audio(self, audio: np.ndarray, sr: int, window_size: float, step_size: float,
                           modes: List[str] = None, stack_data: bool = False) -> np.ndarray:
        # Check for known modes (use mels as default)
        if modes is None:
            modes = ['mels']
        else:
            modes = [m for m in modes if m in self.available_modes]
        if len(modes) == 0:
            modes = ['mels']

        # Determine length of audio files and create empty feature list
        snippet_length = int(window_size * sr)
        features = []
        # Iterate over time windows using the given step size
        for timestamp in np.arange(0.0, int(len(audio) / sr), step_size):
            # Select relevant audio
            start_point = int(timestamp * sr)
            relevant_audio = audio[start_point:start_point + snippet_length]
            if len(relevant_audio) != snippet_length:
                break
            # Calculate features set in modes list
            feat_list = []
            if 'mels' in modes:
                spectrogram = librosa.feature.melspectrogram(y=relevant_audio, sr=sr, n_mels=64)
                spectrogram_db = librosa.power_to_db(S=spectrogram, ref=np.max)
                if self.standardize:
                    feat_list.append((spectrogram_db - np.mean(spectrogram_db)) / np.std(spectrogram_db))
                else:
                    feat_list.append(spectrogram_db)
            if 'stft' in modes:
                stft = librosa.feature.chroma_stft(y=relevant_audio, sr=sr, n_chroma=64)
                if self.standardize:
                    feat_list.append((stft - np.mean(stft)) / np.std(stft))
                else:
                    feat_list.append(stft)
            if 'cqt' in modes:
                cqt = librosa.feature.chroma_cqt(y=relevant_audio, sr=sr, n_chroma=64, bins_per_octave=64)
                if self.standardize:
                    feat_list.append((cqt - np.mean(cqt)) / np.std(cqt))
                else:
                    feat_list.append(cqt)
            # Stack features to 3d-array
            features.append(np.dstack(feat_list))

        return np.array(features)


class WaveFeatures(FeatureEngineering):

    def __init__(self, audio_path: pathlib.Path, label_path: pathlib.Path, audio_sr: int = 22050,
                 label_mode: str = 'max', standardize: bool = False):
        super().__init__(audio_path, label_path, audio_sr, label_mode, standardize)

    def features_for_audio(self, audio: np.ndarray, sr: int, window_size: float, step_size: float,
                           modes: List[str] = None, stack_data: bool = False) -> np.ndarray:
        # Determine length of audio files and create empty feature list
        snippet_length = int(window_size * sr)
        features = []
        # Iterate over time windows using the given step size
        for timestamp in np.arange(0.0, int(len(audio) / sr), step_size):
            # Select relevant audio
            start_point = int(timestamp * sr)
            relevant_audio = audio[start_point:start_point + snippet_length]
            if len(relevant_audio) != snippet_length:
                break
            # Append audio snippet to feature list
            if self.standardize:
                relevant_audio = (relevant_audio - np.mean(relevant_audio)) / np.std(relevant_audio)
            features.append(np.array(relevant_audio).reshape((-1, 1)) if stack_data else relevant_audio)
        return np.array(features)


class ImageFeatures(FeatureEngineering):

    def __init__(self, audio_path: pathlib.Path, label_path: pathlib.Path, audio_sr: int = 22050,
                 label_mode: str = 'max', image_size: Tuple[int, int] = None):
        self.image_size = image_size
        super().__init__(audio_path, label_path, audio_sr, label_mode)

    def features_for_audio(self, audio: np.ndarray, sr: int, window_size: float, step_size: float,
                           modes: List[str] = None, stack_data: bool = False) -> np.ndarray:
        # Determine length of audio files and create empty feature list
        snippet_length = int(window_size * sr)
        features = []
        # Iterate over time windows using the given step size
        for timestamp in np.arange(0.0, int(len(audio) / sr), step_size):
            # Select relevant audio
            start_point = int(timestamp * sr)
            relevant_audio = audio[start_point:start_point + snippet_length]
            if len(relevant_audio) != snippet_length:
                break
            # Create new figure and plot the waveform audio
            fig, axs = plt.subplots(frameon=False)
            axs.plot(relevant_audio)
            axs.set_xticks([])
            axs.set_yticks([])
            plt.tight_layout()
            # Extract the image and append to feature list
            img = self.get_img_from_fig(fig=fig)
            features.append(img)
            plt.close(fig)
        return np.array(features)

    def get_img_from_fig(self, fig: plt.Figure, dpi: int = 180):
        """
        Taken from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        if self.image_size is not None:
            img = cv2.resize(img, self.image_size)
        return img
