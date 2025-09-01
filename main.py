import cv2

from src.camera import Webcam
from src.model import YOLO_detect


def main():

    camera = Webcam(mirror=True, camera_index=1)
    model = YOLO_detect("models/yolo11n.pt") 

    while True:

        frame = camera.read()
        pred = model.predict(frame)

        # Show the frame
        cv2.imshow('Object Detection', pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()