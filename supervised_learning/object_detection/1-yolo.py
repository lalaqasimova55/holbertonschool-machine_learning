#!/usr/bin/env python3
"""YOLO v3 Object Detection"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """YOLO v3 object detector"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """Class constructor"""

        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Calculates the sigmoid of x"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet outputs.

        Args:
            outputs: list of prediction tensors
            image_size: numpy array (image_height, image_width)

        Returns:
            (boxes, box_confidences, box_class_probs)
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h, image_w = image_size

        for i, output in enumerate(outputs):

            grid_h, grid_w, anchor_boxes = output.shape[:3]

            # Network outputs
            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Create grid
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)

            cx, cy = np.meshgrid(cx, cy)

            cx = np.expand_dims(cx, axis=-1)
            cy = np.expand_dims(cy, axis=-1)

            # Predicted center
            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            # Predicted width/height
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = (np.exp(tw) * pw) / input_w
            bh = (np.exp(th) * ph) / input_h

            # Convert to image coordinates
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            # Confidence
            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            # Class probabilities
            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs
