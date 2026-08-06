#!/usr/bin/env python3
"""
YOLOv3 model çıktılarını işlemek için process_outputs metodunu içerir.
"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    YOLOv3 algoritmasını kullanarak nesne tespiti yapan sınıf.
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
            outputs: Darknet modelinden alınan tahminlerin bulunduğu list (ndarray).
            image_size: Orijinal görüntünün boyutu [image_height, image_width].

        Döndürür:
            (boxes, box_confidences, box_class_probs) şeklinde bir tuple.
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
            # t_x, t_y için Sigmoit
            t_xy = 1 / (1 + np.exp(-output[..., 0:2]))
            
            # Box Confidence için Sigmoit
            box_confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(box_confidence)

            # Sınıf olasılıkları için Sigmoit
            box_class_prob = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(box_class_prob)

            # 2. Grid (cx, cy) Koordinat Matrixlerinin Oluşturulması
            grid_x, grid_y = np.meshgrid(np.arange(grid_width), np.arange(grid_height))
            grid_xy = np.stack((grid_x, grid_y), axis=-1)
            grid_xy = np.expand_dims(grid_xy, axis=2)  # Shape: (grid_height, grid_width, 1, 2)

            # 3. Bounding Box Merkez ve Boyutlarının Hesaplanması
            # b_x = sigmoid(t_x) + c_x, b_y = sigmoid(t_y) + c_y
            bx_by = t_xy + grid_xy
            
            # Grid hücre sayısına bölerek [0, 1] aralığına normalize etme
            bx_by[..., 0] /= grid_width
            bx_by[..., 1] /= grid_height

            # b_w = p_w * e^(t_w), b_h = p_h * e^(t_h)
            t_wh = output[..., 2:4]
            anchor_pw_ph = self.anchors[i]  # Shape: (anchor_boxes, 2)
            bw_bh = anchor_pw_ph * np.exp(t_wh)
            
            # Model giriş boyutuna (input_width, input_height) bölerek normalize etme
            bw_bh[..., 0] /= input_width
            bw_bh[..., 1] /= input_height

            # 4. (x1, y1, x2, y2) Koordinatlarının Orijinal Görsele Ölçeklenmesi
            # Merkez ve boyuttan köşe koordinatlarına geçiş
            x1 = (bx_by[..., 0] - (bw_bh[..., 0] / 2)) * image_width
            y1 = (bx_by[..., 1] - (bw_bh[..., 1] / 2)) * image_height
            x2 = (bx_by[..., 0] + (bw_bh[..., 0] / 2)) * image_width
            y2 = (bx_by[..., 1] + (bw_bh[..., 1] / 2)) * image_height

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

        return boxes, box_confidences, box_class_probs
