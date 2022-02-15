import os
import pathlib
from Preprocessing.AudioProcessor import AudioProcessor
from Preprocessing.VideoProcessor import VideoProcessor

# Set paths and constants
VIDEO_PATH = pathlib.Path('data/Video')
AUDIO_PATH = pathlib.Path('data/Audio')
LABEL_PATH = pathlib.Path('data/Labels')
TARGET_SR = 22050
MODEL_TYPE = 'tiny'
WINDOW_SIZE = 0.2
SHOW_VIDEO = True

# Create processing objects
audio_engineer = AudioProcessor(video_path=VIDEO_PATH, audio_path=AUDIO_PATH, target_sr=TARGET_SR)
video_engineer = VideoProcessor(video_path=VIDEO_PATH, label_path=LABEL_PATH, window_size=WINDOW_SIZE,
                                model_type=MODEL_TYPE)


def process_file(file: str):
    filename = file.split('.')[0]
    # Check if file has been processed before and skip in that case
    if f'{filename}.csv' in os.listdir(LABEL_PATH) and f'{filename}.wav' in os.listdir(AUDIO_PATH):
        print(f'Skipping file {filename}!')
        return
    video_engineer.process_file(filename=filename, show_video=SHOW_VIDEO)
    audio_engineer.process_file(filename=filename)


def run_preprocessing():
    files = [f for f in os.listdir(VIDEO_PATH) if f.endswith('.mp4')]
    total_files = len(files)
    count = 1
    for file in files:
        print(f'Processing file {file} ({count}/{total_files})')
        process_file(file=file)
        count += 1


if __name__ == '__main__':
    run_preprocessing()
