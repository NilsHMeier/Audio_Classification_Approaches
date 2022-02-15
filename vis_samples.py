from Utils import Visualizer
from Preprocessing.AudioProcessor import AudioProcessor
import os
import pathlib

AUDIO_PATH = pathlib.Path('data/Audio')
FILES = ['Sample_15.wav']

FILES = FILES if len(FILES) > 0 else os.listdir(AUDIO_PATH)
audio_engineer = AudioProcessor(video_path=None, audio_path=AUDIO_PATH)

for audio_file in FILES:
    # Load the audio file and plot waveform and spectrogram
    y, sr = audio_engineer.load_audio_from_wav(audio_name=audio_file)
    Visualizer.plot_audio(audio=y, sampling_rate=sr, modes=['waveplot', 'melspectrogram', 'chroma_stft'],
                          title=f'Audio Visualization of {audio_file.split(".")[0]}')
