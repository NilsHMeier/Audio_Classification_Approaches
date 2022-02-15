import datetime
import os
import pathlib
import cv2
from Utils import Visualizer
from ComputerVision.ObjectDetection import FrameDifferencing, YoloDetector
from Preprocessing.AudioProcessor import AudioProcessor

VIDEO_DIRECTORY = pathlib.Path('data/Video')
AUDIO_DIRECTORY = pathlib.Path('data/Audio')
VIDEO_NAME = 'Sample_15.mp4'


def play_video(video_path: str):
    # Create capture object and detectors
    video = cv2.VideoCapture(video_path)
    diff_detector = FrameDifferencing(bin_threshold=100, min_area=10000)
    yolo_detector = YoloDetector(model_type='tiny')
    # Read video properties and calculate statistics
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1.0 / fps
    play_time = 0.0
    seconds = int(frames / fps)
    video_time = str(datetime.timedelta(seconds=seconds))
    print(f'Frames={frames} - FPS={fps} - Video-time={video_time}')

    # Create initial objects for frame differencing based detection and start reading frames
    previous_frame = None
    boxes_diff = None
    while True:
        success, frame = video.read()
        if not success:
            print('End of video')
            break
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break

        # Get bounding boxes using yolo
        classes, scores, boxes_yolo = yolo_detector.detect_objects(frame=frame, vehicles_only=True)
        yolo_frame = frame.copy()
        for x, y, w, h in boxes_yolo:
            cv2.rectangle(img=yolo_frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=3)
        cv2.putText(img=yolo_frame, text=str(datetime.timedelta(seconds=play_time)), org=(15, 15),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0))
        cv2.imshow('YOLO Detection', yolo_frame)

        # Get bounding boxes using frame differencing
        if previous_frame is not None:
            boxes_diff = diff_detector.detect_objects(previous_frame=previous_frame, recent_frame=frame)
        previous_frame = frame.copy()
        if boxes_diff is not None:
            for x, y, w, h in boxes_diff:
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=3)
        cv2.putText(img=frame, text=str(datetime.timedelta(seconds=play_time)), org=(15, 15),
                    fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0))
        cv2.imshow('Frame Differencing', frame)
        play_time += time_per_frame

    video.release()
    cv2.destroyAllWindows()


def visualize_audio():
    # Extract audio from video file and plot it in different modes
    audio_engineer = AudioProcessor(video_path=VIDEO_DIRECTORY, audio_path=AUDIO_DIRECTORY)
    audio, sr = audio_engineer.extract_audio_from_video(video_name=VIDEO_NAME.split('.')[0])
    Visualizer.plot_audio(audio=audio, sampling_rate=sr,
                          modes=['waveplot', 'chroma_cqt', 'chroma_stft', 'melspectrogram'])


def main():
    visualize_audio()
    play_video(os.path.join(VIDEO_DIRECTORY, VIDEO_NAME))


if __name__ == '__main__':
    main()
