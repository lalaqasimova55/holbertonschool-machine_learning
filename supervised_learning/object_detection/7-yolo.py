#!/usr/bin/env python3
"""
YOLOv3 object detection end-to-end pipeline: image loading, preprocessing,
model prediction, post-processing (NMS), and visual box display.
"""
import cv2
import glob
import numpy as np
import os
import tensorflow.keras as K


class Yolo:
    """
    Class that performs object detection using the YOLOv3 algorithm.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor for Yolo class.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes Darknet model outputs.

        Parameters:
            outputs: list of numpy.ndarrays containing predictions.
            image_size: numpy.ndarray containing [image_height, image_width].

        Returns:
            (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = 1 / (1 + np.exp(-output[..., 0:2]))

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            grid_x, grid_y = np.meshgrid(
                np.arange(grid_width),
                np.arange(grid_height)
            )
            grid_xy = np.stack((grid_x, grid_y), axis=-1)
            grid_xy = np.expand_dims(grid_xy, axis=2)

            bx_by = t_xy + grid_xy
            bx_by[..., 0] /= grid_width
            bx_by[..., 1] /= grid_height

            t_wh = output[..., 2:4]
            anchor_pw_ph = self.anchors[i]
            bw_bh = anchor_pw_ph * np.exp(t_wh)

            bw_bh[..., 0] /= input_width
            bw_bh[..., 1] /= input_height

            x1 = (bx_by[..., 0] - (bw_bh[..., 0] / 2)) * image_width
            y1 = (bx_by[..., 1] - (bw_bh[..., 1] / 2)) * image_height
            x2 = (bx_by[..., 0] + (bw_bh[..., 0] / 2)) * image_width
            y2 = (bx_by[..., 1] + (bw_bh[..., 1] / 2)) * image_height

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes by threshold score.

        Parameters:
            boxes: list of processed boundary boxes.
            box_confidences: list of processed box confidences.
            box_class_probs: list of processed box class probabilities.

        Returns:
            (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            mask = box_score >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-Max Suppression to filter duplicate overlapping boxes.

        Parameters:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes: numpy.ndarray of shape (?,)
            box_scores: numpy.ndarray of shape (?,)

        Returns:
            (box_predictions, predicted_box_classes, predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            order = np.argsort(cls_scores)[::-1]
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            while len(cls_boxes) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[0])

                if len(cls_boxes) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                intersection_w = np.maximum(0.0, x2 - x1)
                intersection_h = np.maximum(0.0, y2 - y1)
                intersection_area = intersection_w * intersection_h

                box0_area = (cls_boxes[0, 2] - cls_boxes[0, 0]) * (
                    cls_boxes[0, 3] - cls_boxes[0, 1]
                )
                boxes_area = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (
                    cls_boxes[1:, 3] - cls_boxes[1:, 1]
                )

                union_area = box0_area + boxes_area - intersection_area
                iou = intersection_area / union_area

                below_threshold = np.where(iou <= self.nms_t)[0]

                cls_boxes = cls_boxes[below_threshold + 1]
                cls_scores = cls_scores[below_threshold + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        Loads images from a specified folder path.

        Parameters:
            folder_path: string representing path to the folder

        Returns:
            (images, image_paths)
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(image_path) for image_path in image_paths]

        return images, image_paths

    def preprocess_images(self, images):
        """
        Preprocesses images for the Darknet model.

        Parameters:
            images: list of images as numpy.ndarrays

        Returns:
            (pimages, image_shapes)
        """
        input_h = self.model.input.shape[2]
        input_w = self.model.input.shape[1]

        pimages = []
        image_shapes = []

        for img in images:
            h, w, _ = img.shape
            image_shapes.append((h, w))

            resized_img = cv2.resize(
                img,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            rescaled_img = resized_img / 255.0
            pimages.append(rescaled_img)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays an image with all boundary boxes, class names, and box scores.

        Parameters:
            image: numpy.ndarray containing an unprocessed image
            boxes: numpy.ndarray containing boundary boxes for image
            box_classes: numpy.ndarray containing class indices for each box
            box_scores: numpy.ndarray containing box scores for each box
            file_name: file path where original image is stored
        """
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box)

            cls_idx = box_classes[i]
            score = box_scores[i]
            class_name = self.class_names[cls_idx]
            label = f"{class_name} {score:.2f}"

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key & 0xFF == ord('s'):
            if not os.path.exists('detections'):
                os.makedirs('detections')
            save_path = os.path.join('detections', file_name)
            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Displays all images with bounding boxes and predicts detections.

        Parameters:
            folder_path: string representing path to the folder of images

        Returns:
            (predictions, image_paths):
                predictions: list of tuples (boxes, box_classes, box_scores)
                image_paths: list of image paths corresponding to predictions
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i in range(len(images)):
            # Extract output predictions corresponding to image i
            image_outputs = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs,
                image_shapes[i]
            )

            filtered_boxes, box_classes, box_scores = self.filter_boxes(
                boxes,
                box_confidences,
                box_class_probs
            )

            b, c, s = self.non_max_suppression(
                filtered_boxes,
                box_classes,
                box_scores
            )

            predictions.append((b, c, s))

            # Extract image filename without directory path for window title
            file_name = os.path.basename(image_paths[i])
            self.show_boxes(images[i], b, c, s, file_name)

        return predictions, image_paths
