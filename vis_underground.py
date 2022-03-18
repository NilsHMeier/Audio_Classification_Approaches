from pathlib import Path
import numpy as np
from Preprocessing.AudioProcessor import AudioProcessor
import matplotlib.pyplot as plt
import librosa.display

# Set sampling rate
sr = 22050
audio_engineer = AudioProcessor(video_path=None, audio_path=Path('data/Audio'), target_sr=sr)
# Load and crop samples
asphalt_audio, _ = audio_engineer.load_audio_from_wav(audio_name='Sample_25')
asphalt_audio = asphalt_audio[48*sr:54*sr]
pavement_audio, _ = audio_engineer.load_audio_from_wav(audio_name='Pflaster_Sample')
pavement_audio = pavement_audio[2*sr:8*sr]

# Create plots
fig, axs = plt.subplots(2, 2, figsize=(16, 6))
for i, (ground, audio) in enumerate(zip(['Asphalt', 'Cobblestone'], [asphalt_audio, pavement_audio])):
    axs[0][i].plot(np.arange(len(audio)) / sr, audio)
    axs[0][i].set(title=f'Waveform of {ground}', xlim=(0, len(audio)/sr))
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, hop_length=64)
    librosa.display.specshow(data=librosa.power_to_db(spectrogram), x_axis='time', y_axis='log', sr=sr, ax=axs[1][i])
    axs[1][i].set(title=f'Melspectrogram of {ground}')
plt.show()
