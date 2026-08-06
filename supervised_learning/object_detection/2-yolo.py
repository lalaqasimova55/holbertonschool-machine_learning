#!/usr/bin/env python3
"""
YOLOv3 model çıxışlarını emal etmək və filtrdən keçirmək üçün Yolo sinfi.
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    YOLOv3 alqoritmi ilə obyekt tespiti edən sinif.
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo sinfinin kurucu metodu.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Darknet modelindən gələn təxmin çıxışlarını emal edir.

        Parametrlər:
            outputs: Model təxmin dizisi.
            image_size: Orijinal şəkil ölçüsü [image_height, image_width].

        Qaytarır:
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

            # 1. Sigmoid Aktivasiyası İşlemləri
            t_xy = 1 / (1 + np.exp(-output[..., 0:2]))

            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            # 2. Grid (cx, cy) Koordinat Matrislərinin Yaradılması
            grid_x, grid_y = np.meshgrid(
                np.arange(grid_width),
                np.arange(grid_height)
            )
            grid_xy = np.stack((grid_x, grid_y), axis=-1)
            grid_xy = np.expand_dims(grid_xy, axis=2)

            # 3. Bounding Box Mərkəz və Ölçülərinin Hesablanması
            bx_by = t_xy + grid_xy

            bx_by[..., 0] /= grid_width
            bx_by[..., 1] /= grid_height

            t_wh = output[..., 2:4]
            anchor_pw_ph = self.anchors[i]
            bw_bh = anchor_pw_ph * np.exp(t_wh)

            bw_bh[..., 0] /= input_width
            bw_bh[..., 1] /= input_height

            # 4. Koordinatların Orijinal Şəkilə Ölçəklənməsi
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
            # Hər bir qutu və sinif üçün ümumi skorun hesablanması
            scores = box_confidences[i] * box_class_probs[i]

            # Ən yüksək skora malik sinfin indeksi və skor dəyəri
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            # Threshold-a əsasən maskalanma (skoru class_t-dən böyük olanlar)
            mask = box_score >= self.class_t

            # Filter olunmuş məlumatların siyahıya əlavə olunması
            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_class[mask])
            box_scores.append(box_score[mask])

        # Bütün miqyaslardan gələn məlumatların birleşdirilməsi
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
