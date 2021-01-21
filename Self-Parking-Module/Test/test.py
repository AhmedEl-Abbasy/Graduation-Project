"""
Yolov4 is a library that can be installed using pip install yolov4 . just download yolov4.weights
"""
from yolov4.tf import YOLOv4
import cv2 as cv

yolo = YOLOv4()

yolo.classes = "coco.names"
yolo.input_size = (640, 480)

yolo.make_model()
yolo.load_weights("yolov4.weights", weights_type="yolo")

yolo.inference("trial.jpg")