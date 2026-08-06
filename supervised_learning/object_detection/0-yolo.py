#!/usr/bin/env python3
"""YOLO v3 Object Detection"""


import tensorflow.keras as K


class Yolo:
    """YOLO v3 object detector"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path: path to the Darknet Keras model
            classes_path: path to the file containing class names
            class_t: box score threshold
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of anchor boxes
        """

        # Load the trained YOLO model
        self.model = K.models.load_model(model_path)

        # Load class names
        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        # Store thresholds and anchors
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
