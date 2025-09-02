from itertools import cycle
import cv2

from src.camera import Webcam
from src.model import YOLO_detect, YOLO_pose

def main():

    camera = Webcam(mirror=False, camera_index=1)
    models = [
        YOLO_detect("models/yolo11n.pt"),
        YOLO_pose("models/yolo11n-pose.pt")
    ]
    model_cycle = cycle(models)
    model = next(model_cycle)

    while True:

        frame = camera.read()
        pred = model.predict(frame)

        # Show the frame
        cv2.imshow('Object Detection', pred)

        # read keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('n'):
            model = next(model_cycle)
 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()