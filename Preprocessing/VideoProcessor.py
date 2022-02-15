import pathlib
import os
import datetime
from typing import List, Tuple
import cv2
import numpy as np
import pandas as pd
from ComputerVision.ObjectDetection import YoloDetector
from tqdm import tqdm


class VideoProcessor:

    def __init__(self, video_path: pathlib.Path, label_path: pathlib.Path, window_size: float = 0.5,
                 model_type: str = 'normal'):
        self.video_path = video_path
        self.label_path = label_path
        self.window_size = window_size
        self.detector = YoloDetector(model_type=model_type)

    def process_file(self, filename: str, show_video: bool = False):
        label_df = self.run_labeling_for_file(filename=filename, show_video=show_video)
        label_df.to_csv(path_or_buf=os.path.join(self.label_path, f'{filename}.csv'))

    def run_labeling_for_file(self, filename: str, show_video: bool = False) -> pd.DataFrame:
        print(f'Running labeling for file {filename}')
        # Create capture object and load frames
        capture = cv2.VideoCapture(os.path.join(self.video_path, f'{filename}.mp4'))
        frames, timestamps = self.load_frames(capture=capture)
        # Determine center box
        center_box = self.determine_center_box(width=capture.get(propId=cv2.CAP_PROP_FRAME_WIDTH),
                                               height=capture.get(propId=cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        # Create dataframe for labeling
        label_df = self.create_label_dataframe(video_length=np.max(timestamps))

        car_in_center = False
        for i in tqdm(range(len(frames))):
            frame, timestamp = frames[i], timestamps[i]
            # Detect objects using yolo
            classes, scores, boxes = self.detector.detect_objects(frame, vehicles_only=True)
            # Check if car is in center and change label if car has left center box
            for box in boxes:
                if self.is_centroid_in_center(bounding_box=box, center_box=center_box):
                    car_in_center = True
                    break
                if car_in_center:
                    car_in_center = False
                    label_df.at[int(timestamp // self.window_size), 'label'] = 1

            if show_video:
                cv2.rectangle(frame, center_box, (0, 255, 0), 3)
                cv2.putText(img=frame, text=str(datetime.timedelta(seconds=timestamp)), org=(15, 15),
                            fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0))
                for (classid, score, box) in zip(classes, scores, boxes):
                    color = (0, 0, 255) if self.is_centroid_in_center(box, center_box) else (0, 255, 0)
                    label = '%s : %f' % (self.detector.class_names[classid[0]], score)
                    cv2.rectangle(frame, box, color, 2)
                    centroid_x, centroid_y = box[0] + int(box[2] / 2), box[1] + int(box[3] / 2)
                    cv2.circle(frame, (centroid_x, centroid_y), radius=0, color=(0, 0, 255), thickness=5)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.imshow('Detections', frame)
                cv2.waitKey(1)

        if show_video:
            cv2.destroyAllWindows()
        del frames

        return label_df

    def create_label_dataframe(self, video_length: float) -> pd.DataFrame:
        timestamps = np.arange(0, video_length, self.window_size).round(decimals=5)
        df = pd.DataFrame(data={'time': timestamps})
        df['label'] = 0
        return df

    @staticmethod
    def load_frames(capture: cv2.VideoCapture) -> Tuple[List[np.ndarray], np.ndarray]:
        frames = []
        time_per_frame = 1.0 / capture.get(propId=cv2.CAP_PROP_FPS)
        ret, frame = capture.read()
        while ret:
            frames.append(frame)
            ret, frame = capture.read()
        timestamps = np.arange(0, len(frames) * time_per_frame, time_per_frame)
        return frames, timestamps

    @staticmethod
    def determine_center_box(width: int, height: int, box_percentage: float = 0.1) -> Tuple[int, int, int, int]:
        x = int((width * (1 - box_percentage)) / 2)
        y = int(0)
        w = int(box_percentage * width)
        h = int(height)
        return x, y, w, h

    @staticmethod
    def is_centroid_in_center(bounding_box: Tuple, center_box: Tuple) -> bool:
        # Determine centroid of boundig box and check if it is inside of center box
        centroid_x, centroid_y = bounding_box[0] + int(bounding_box[2] / 2), bounding_box[1] + int(bounding_box[3] / 2)
        return center_box[0] < centroid_x < center_box[0] + center_box[2] and \
               center_box[1] < centroid_y < center_box[1] + center_box[3]
