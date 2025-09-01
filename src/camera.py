from abc import ABC, abstractmethod

import numpy as np
import cv2


class Camera(ABC):

    @abstractmethod
    def read(self) -> np.ndarray:
        return


class Webcam(Camera):
    def __init__(self, camera_index: int  = 0, mirror: bool = False):
        super().__init__()
        self.cap_device = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        self.mirror = mirror

        assert self.cap_device.isOpened(),"Failed to open capture device."

    def __del__(self):
        if hasattr(self, 'cap_device') and self.cap_device.isOpened():
            self.cap_device.release()

    def read(self):
        ret, frame = self.cap_device.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera")
        
        if self.mirror: 
           # flip image horizontally
           frame = cv2.flip(frame, flipCode=1)

        return frame


if __name__ == '__main__':

    cam = Webcam()

    while True:
        frame = cam.read()

        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


 
    