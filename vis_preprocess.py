import cv2
from Utils import Visualizer
from ComputerVision.ObjectDetection import YoloDetector, FrameDifferencing
from Preprocessing.VideoProcessor import VideoProcessor

# Create detector object
yolo_detector = YoloDetector(model_type='normal')
diff_detector = FrameDifferencing(bin_threshold=100, min_area=10000)


def visualize_detection():
    # Load exemplary images
    frame1 = cv2.imread('data/Images/Detection_1.JPG')
    frame2 = cv2.imread('data/Images/Detection_2.JPG')

    # Run yolo detection and draw boxes
    classes, scores, boxes = yolo_detector.detect_objects(frame=frame1, vehicles_only=True)
    yolo_frame = frame1.copy()
    for box in boxes:
        cv2.rectangle(yolo_frame, box, (0, 0, 255), 2)
        centroid_x, centroid_y = box[0] + int(box[2] / 2), box[1] + int(box[3] / 2)
        cv2.circle(yolo_frame, (centroid_x, centroid_y), radius=0, color=(0, 0, 255), thickness=5)

    # Run frame differencing detection and draw boxes
    boxes = diff_detector.detect_objects(previous_frame=frame1, recent_frame=frame2)
    diff_frame = frame2.copy()
    for box in boxes:
        cv2.rectangle(diff_frame, box, (0, 0, 255), 2)
        centroid_x, centroid_y = box[0] + int(box[2] / 2), box[1] + int(box[3] / 2)
        cv2.circle(diff_frame, (centroid_x, centroid_y), radius=0, color=(0, 0, 255), thickness=5)

    # Display detections
    Visualizer.display_images(images=[yolo_frame, diff_frame],
                              titles=['YOLO Detections', 'Frame Differencing Detections'], plot_shape=(1, 2),
                              title='Object Detection')


def visualize_labeling():
    # Load frames
    before_center = cv2.imread('data/Images/Labeling_1.JPG')
    in_center = cv2.imread('data/Images/Labeling_2.JPG')
    after_center = cv2.imread('data/Images/Labeling_3.JPG')

    # Determine center box
    center_box = VideoProcessor.determine_center_box(width=before_center.shape[1], height=before_center.shape[0],
                                                     box_percentage=0.1)

    # Detect cars in each frame using yolo and check if centroid of car is in center box
    for frame in [before_center, in_center, after_center]:
        classes, scores, boxes = yolo_detector.detect_objects(frame, vehicles_only=True)
        cv2.rectangle(frame, center_box, (0, 255, 0), 3)
        for (class_id, score, box) in zip(classes, scores, boxes):
            color = (0, 0, 255) if VideoProcessor.is_centroid_in_center(box, center_box) else (0, 255, 0)
            label = '%s : %f' % (yolo_detector.class_names[class_id[0]], score)
            cv2.rectangle(frame, box, color, 2)
            centroid_x, centroid_y = box[0] + int(box[2] / 2), box[1] + int(box[3] / 2)
            cv2.circle(frame, (centroid_x, centroid_y), radius=0, color=(0, 0, 255), thickness=5)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display images
    Visualizer.display_images(images=[before_center, in_center, after_center],
                              titles=['Before Center', 'In Center', 'After Center'], plot_shape=(1, 3),
                              title='Labeling Process')


if __name__ == '__main__':
    visualize_detection()
    visualize_labeling()
