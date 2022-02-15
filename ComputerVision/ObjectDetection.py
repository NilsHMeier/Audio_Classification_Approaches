import pathlib
import copy
import cv2
import numpy as np
from typing import List, Tuple


class YoloDetector:
    CONFIDENCE_THRESHOLD = 0.7
    NMS_THRESHOLD = 0.4
    VEHICLES = ['car', 'motorcycle', 'bus', 'truck']

    def __init__(self, model_type: str = 'normal'):
        if model_type == 'normal':
            weights_path = pathlib.Path('ComputerVision/YOLO/yolov4.weights')
            config_path = pathlib.Path('ComputerVision/YOLO/yolov4.cfg')
        else:
            weights_path = pathlib.Path('ComputerVision/YOLO/yolov4-tiny.weights')
            config_path = pathlib.Path('ComputerVision/YOLO/yolov4-tiny.cfg')
        net = cv2.dnn.readNet(weights_path.__str__(), config_path.__str__())
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)
        with open(pathlib.Path('ComputerVision/YOLO/coco-labels.txt'), 'r') as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

    def detect_objects(self, frame: np.ndarray, vehicles_only: bool = True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Use model to get all objects
        classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        # If required filter out all non vehicle objects
        if vehicles_only:
            classes, scores, boxes = self.filter_detections(classes=classes, scores=scores, boxes=boxes)
        # Return bounding boxes, scores and predicted classes
        return classes, scores, boxes

    def filter_detections(self, classes, scores, boxes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = [i for i, c in enumerate(classes) if self.class_names[c[0]] in self.VEHICLES]
        return np.array(classes)[indices], np.array(scores)[indices], np.array(boxes)[indices]


class FrameDifferencing:

    def __init__(self, bin_threshold: int, min_area: int):
        self.bin_threshold = bin_threshold
        self.min_area = min_area

    def detect_objects(self, previous_frame: np.ndarray, recent_frame: np.ndarray):
        # Convert frames to grayscale, calculate absolute difference and apply threshold
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        recent_frame = cv2.cvtColor(recent_frame, cv2.COLOR_BGR2GRAY)
        abs_difference = cv2.absdiff(previous_frame, recent_frame)
        _, binary = cv2.threshold(abs_difference, self.bin_threshold, 255, cv2.THRESH_BINARY)
        # Apply dilation
        dilated = cv2.dilate(binary, kernel=np.ones((3, 3), dtype=np.int8), iterations=15)
        # Find contours and check if minimum size is given
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > self.min_area]
        # Check for nested boxes and unstack if needed
        return self.unstack_boxes(boxes)

    @staticmethod
    def unstack_boxes(boxes: List[Tuple]):
        # Check if a box is completely inside another box
        to_delete = []
        for i, inner_box in enumerate(boxes):
            remaining_boxes = copy.deepcopy(boxes)
            remaining_boxes.pop(i)
            for outer_box in remaining_boxes:
                if FrameDifferencing.is_box_inside(inner_box, outer_box):
                    to_delete.append(i)
                    break
        unstacked_boxes = [boxes[i] for i in range(len(boxes)) if i not in to_delete]
        return unstacked_boxes

    @staticmethod
    def is_box_inside(inner_box: Tuple, outer_box: Tuple) -> bool:
        x1, y1, w1, h1 = outer_box
        x2, y2, w2, h2 = inner_box
        return x2 >= x1 and x2 + w2 <= x1 + w1 and y2 >= y1 and y2 + h2 <= y1 + h1
