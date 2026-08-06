#!/usr/bin/env python3
"""
YOLOv3 model çıktılarını işlemek, NMS uygulamak ve görselleri yüklemek.
"""
import cv2
import glob
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    YOLOv3 algoritması ile nesne tespiti yapan sınıf.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo sınıfı için kurucu metod.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Darknet modelinden gelen tahmin çıktılarını işler.

        Parametreler:
            outputs: Model tahmin dizisi.
            image_size: Orijinal görüntü boyutu [image_height, image_width].

        Döndürür:
            (boxes, box_confidences, box_class_probs) tuple.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # 1. Sigmoid Aktivasyonu İşlemleri
            t_xy = 1 / (1 + np.exp(-output[..., 0:2]))

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            # 2. Grid (cx, cy) Koordinat Matrislerinin Oluşturulması
            grid_x, grid_y = np.meshgrid(
                np.arange(grid_width),
                np.arange(grid_height)
            )
            grid_xy = np.stack((grid_x, grid_y), axis=-1)
            grid_xy = np.expand_dims(grid_xy, axis=2)

            # 3. Bounding Box Merkez ve Boyutlarının Hesaplanması
            bx_by = t_xy + grid_xy

            bx_by[..., 0] /= grid_width
            bx_by[..., 1] /= grid_height

            t_wh = output[..., 2:4]
            anchor_pw_ph = self.anchors[i]
            bw_bh = anchor_pw_ph * np.exp(t_wh)

            bw_bh[..., 0] /= input_width
            bw_bh[..., 1] /= input_height

            # 4. Koordinatların Orijinal Görsele Ölçeklenmesi
            x1 = (bx_by[..., 0] - (bw_bh[..., 0] / 2)) * image_width
            y1 = (bx_by[..., 1] - (bw_bh[..., 1] / 2)) * image_height
            x2 = (bx_by[..., 0] + (bw_bh[..., 0] / 2)) * image_width
            y2 = (bx_by[..., 1] + (bw_bh[..., 1] / 2)) * image_height

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Aşağı skorlu boundary box-ları süzür (filtrdən keçirir).

        Parametrlər:
            boxes: processed boundary boxes siyahısı.
            box_confidences: processed box confidences siyahısı.
            box_class_probs: processed box class probabilities siyahısı.

        Qaytarır:
            (filtered_boxes, box_classes, box_scores) tuple.
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
        Non-Max Suppression (NMS) tətbiq edərək təkrarlanan qutuları silir.

        Parametrlər:
            filtered_boxes: numpy.ndarray of shape (?, 4)
            box_classes: numpy.ndarray of shape (?,)
            box_scores: numpy.ndarray of shape (?)

        Qaytarır:
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
        Belirtilen klasördeki tüm görselleri yükler.

        Parametreler:
            folder_path: Yüklenecek görsellerin klasör yolu.

        Döndürür:
            (images, image_paths) şeklinde tuple.
        """
        image_paths = glob.glob(folder_path + '/*', recursive=False)
        images = [cv2.imread(image_path) for image_path in image_paths]

        return images, image_paths
