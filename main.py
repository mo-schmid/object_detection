import argparse
from itertools import cycle
import cv2

from src.camera import Webcam
from src.model import YOLO_detect, YOLO_pose

def main():

    parser = argparse.ArgumentParser(description='Object Detection with YOLO')
    parser.add_argument('-c', '--camera-index', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('-l', '--language', type=str, default='en', help='Language (default: en)')
    args = parser.parse_args()

    camera = Webcam(mirror=False, camera_index=args.camera_index)
    models = [
        YOLO_detect("models/yolo11n.pt", language=args.language),
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