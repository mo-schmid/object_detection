from abc import ABC, abstractmethod

import cv2
import numpy as np
import ultralytics
import translate
import logging

LOGGER = logging.getLogger(__name__)

class Model(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        return

class YOLO_detect(Model):
    def __init__(self, checkpoint: str ="yolo11n.pt", threshold: float = 0.5, language: str = None):
        self.model = ultralytics.YOLO(checkpoint)
        self.threshold = threshold
        self.color_lut = self._create_color_lut()
        self.cls_lut = self._create_cls_lut(language=language)

    def predict(self, image: np.ndarray) -> np.ndarray:

        results = self.model(image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                if conf > self.threshold:
                    color = self.color_lut[cls]
                    label = self.cls_lut[cls]
            
                    pt1 = box[:2]
                    pt2 = box[2:]
                    cv2.rectangle(image, pt1, pt2, color=color, thickness=2)

                    cv2.putText(
                        image,
                        f"{label} {conf:.2f}",
                        (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.5,
                        color=color,
                        thickness=3,
                    )

        return image

    def _create_color_lut(self):
        """Create a color lookup table for all available classes."""
        if hasattr(self.model, 'names') and self.model.names:
            num_classes = len(self.model.names)
            # Generate distinct colors for each class
            colors = []
            for i in range(num_classes):
                # Use HSV color space for better color distribution
                hue = int(180 * i / num_classes)
                color_hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                colors.append(tuple(map(int, color_bgr)))
            return dict(zip(range(num_classes), colors))
        return {}
        
    def _create_cls_lut(self, language: str = None) -> dict:
        if hasattr(self.model, 'names') and self.model.names:
            class_labels = self.model.names.values()

            if language:
                LOGGER.info(f'Translating labels to {language} ...')
                translator = translate.Translator(to_lang=language, from_lang='en')
                class_labels = [translator.translate(lbl) for lbl in class_labels]
        
        return dict(zip(range(len(class_labels)), class_labels))


class YOLO_pose(Model):
    def __init__(self, checkpoint: str ="yolo11n-pose.pt", threshold: float = 0.5, language: str = None):
        self.model = ultralytics.YOLO(checkpoint)
        self.threshold = threshold
        self.lines = {
            "shoulder": (5, 6),
            "arm_upper_l": (5, 7),
            "arm_lower_l": (7, 9),
            "arm_upper_r": (6, 8),
            "arm_lower_r": (8, 10),
            "body_l": (5, 11),
            "body_r": (6, 12),
            "hips": (11, 12),
            "leg_upper_l": (11, 13),
            "leg_lower_l": (13, 15),
            "leg_upper_r": (12, 14),
            "leg_lower_r": (14, 16)
        }
        self.color = (0, 0, 255)
        self.radius = 5
        self.thickness = 5


    def predict(self, image: np.ndarray) -> np.ndarray:

        results = self.model(image)
        for result in results:

            keypoints = result.keypoints.data.cpu().numpy()
            assert keypoints.shape == (1, 17, 3), "Wrong assumption about keypoints shape"

            X = keypoints[0, :, 0].astype(int)
            Y = keypoints[0, :, 1].astype(int)
            conf = keypoints[0, :, 2]

            for x, y, c in zip(X, Y, conf):
                if c > self.threshold:
                    cv2.circle(image, (x, y), self.radius, self.color, -1)

            for i, j in self.lines.values():
                if conf[i] > self.threshold and conf[j] > self.threshold:
                    cv2.line(image, (X[i], Y[i]), (X[j], Y[j]), color=self.color, thickness=self.thickness)

        return image
