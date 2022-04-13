import os
import datetime as dt
import pathlib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from Preprocessing.AudioProcessor import AudioProcessor
from Preprocessing.FeatureEngineering import WaveFeatures
from Utils import Util, Visualizer

FILE_NAMES = ['Low_01.wav', 'Low_02.wav', 'Low_03.wav']
AUDIO_PATH = pathlib.Path('data/Audio')
VIDEO_PATH = pathlib.Path('')
LABEL_PATH = pathlib.Path('')


def predict_file(filename: str, model: Model):
    # Load audio
    audio_engineer = AudioProcessor(video_path=VIDEO_PATH, audio_path=AUDIO_PATH)
    audio, sr = audio_engineer.load_audio_from_wav(audio_name=filename)
    # Calculate features
    feature_engineer = WaveFeatures(audio_path=AUDIO_PATH, label_path=LABEL_PATH)
    features = feature_engineer.features_for_audio(audio=audio, sr=sr, window_size=1.25, step_size=1.25,
                                                   stack_data=True)
    # Predict the features
    y_pred = np.argmax(model.predict(features), axis=1)

    # Aggregate and plot the predictions
    start_time = dt.datetime.fromtimestamp(AUDIO_PATH.joinpath(filename).stat().st_mtime, tz=dt.timezone.utc)
    aggregated_predictions = Util.aggregate_predictions(predictions=y_pred, step_size=1.25, aggregation_period=30,
                                                        start_time=start_time)
    Visualizer.plot_aggregated_predictions(predictions_df=aggregated_predictions, suptitle=filename)


def main():
    # Load model
    model = load_model('models/Wave/WaveNet.h5', compile=False)
    # Predict all files
    for file in FILE_NAMES:
        predict_file(filename=file, model=model)


if __name__ == '__main__':
    main()
