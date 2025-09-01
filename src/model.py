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

class YOLO_detect():
    def __init__(self, checkpoint: str ="yolo11n.pt", threshold: float = 0.5, language: str = None):
        self.model = ultralytics.YOLO(checkpoint)
        self.threshold = threshold
        self.color_lut = self._create_color_lut()
        self.cls_lut = self._create_cls_lut(language=language)

    def predict(self, image: np.ndarray) -> np.ndarray:

        results = self.model(image)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
            confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
            clss = result.boxes.cls.cpu().numpy() if hasattr(result.boxes, 'cls') else []
            for box, conf, cls in zip(boxes, confs, clss):
                if conf > self.threshold:
                    cls_idx = int(cls)

                    color = self.color_lut[cls_idx]
                    label = self.cls_lut[cls_idx]
              
                    x1, y1, x2, y2 = map(int, box)

                    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=2)
                    cv2.putText(
                        image,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
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


