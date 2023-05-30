import argparse
import cv2
import csv
from lib import camera, BreakException
from typing import Tuple

def generate_training_set(image: cv2.Mat, hand_detect: Tuple[bool, list[float]]):
    global isCapture, i, writer, class_name

    has_hand, distances = hand_detect
    cv2.putText(image, f'{i}/{args.iter}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if isCapture and has_hand and i < args.iter:
        distances.append(class_name)
        writer.writerow(distances)
        i += 1

    cv2.imshow("collect", image)

    match cv2.waitKey(5) & 0xFF:
        case 115:
            isCapture = True
        case 27:
            raise BreakException("Shutdown...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training data generator')
    parser.add_argument('name', type=str)
    parser.add_argument('-i', '--iter', type=int, default=100)

    args = parser.parse_args()
    i = 0
    isCapture = False
    class_name = args.name

    with open('train.csv', 'a') as file:
        writer = csv.writer(file)
        camera(generate_training_set)
