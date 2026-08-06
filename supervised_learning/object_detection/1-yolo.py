#!/usr/bin/env python3
"""Yolo Object Detection"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Class constructor
        """

        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs

        Returns:
            boxes, box_confidences, box_class_probs
        """

        boxes = []
        box_confidences = []
        box_class_probs = []

        # Input dimensions of the network
        _, input_h, input_w, _ = self.model.input.shape.as_list()

        image_h, image_w = image_size

        for output_idx, output in enumerate(outputs):

            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            # Grid coordinates
            cx = np.arange(grid_w)
            cy = np.arange(grid_h)

            cx, cy = np.meshgrid(cx, cy)

            cx = np.tile(cx[..., np.newaxis], (1, 1, anchor_boxes))
            cy = np.tile(cy[..., np.newaxis], (1, 1, anchor_boxes))

            # Center coordinates
            bx = (self.sigmoid(tx) + cx) / grid_w
            by = (self.sigmoid(ty) + cy) / grid_h

            # Width and height
            anchor_w = self.anchors[output_idx, :, 0]
            anchor_h = self.anchors[output_idx, :, 1]

            bw = (np.exp(tw) * anchor_w) / input_w
            bh = (np.exp(th) * anchor_h) / input_h

            # Convert to original image coordinates
            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

            boxes.append(np.stack((x1, y1, x2, y2), axis=-1))

            # Object confidence
            box_confidences.append(self.sigmoid(output[..., 4:5]))

            # Class probabilities
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs
