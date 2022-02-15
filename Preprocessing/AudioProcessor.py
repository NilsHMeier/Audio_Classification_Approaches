import os
from typing import Tuple
import librosa
import moviepy.editor as mp_editor
import pathlib
import numpy as np
import soundfile


class AudioProcessor:

    def __init__(self, video_path: pathlib.Path, audio_path: pathlib.Path, target_sr: int = 22050):
        self.video_path = video_path
        self.audio_path = audio_path
        self.target_sr = target_sr

    def process_file(self, filename: str):
        audio, sr = self.extract_audio_from_video(video_name=filename)
        self.save_audio(audio=audio, sr=sr, filename=filename)

    def extract_audio_from_video(self, video_name: str) -> Tuple[np.ndarray, int]:
        video_path = os.path.join(self.video_path, f'{video_name}.mp4')
        temp_path = os.path.join(self.audio_path, f'{video_name}_temp.wav')
        audio_clip = mp_editor.AudioFileClip(video_path)
        audio_clip.write_audiofile(filename=temp_path, logger=None)
        audio, sr = librosa.load(path=temp_path, sr=self.target_sr)
        os.remove(path=temp_path)
        return audio, sr

    def load_audio_from_wav(self, audio_name: str) -> Tuple[np.ndarray, int]:
        audio_name = audio_name if audio_name.endswith('.wav') else f'{audio_name}.wav'
        audio_path = os.path.join(self.audio_path, audio_name)
        audio, sr = librosa.load(path=audio_path, sr=self.target_sr)
        return audio, sr

    def resample_audio(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
        return librosa.resample(y=audio, sr=sr, target_sr=self.target_sr), self.target_sr

    def save_audio(self, audio: np.ndarray, sr: int, filename: str):
        filename = filename if filename.endswith('.wav') else f'{filename}.wav'
        file_path = os.path.join(self.audio_path, filename)
        soundfile.write(file=file_path, data=audio, samplerate=sr)
