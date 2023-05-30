import cv2
import numpy as np
from math import sqrt
from mediapipe.python.solutions import hands
from typing import Tuple, Callable


class BreakException(Exception):
    pass


class Camera():
    def __init__(self) -> None:
        self.vid = cv2.VideoCapture(0)

    def __enter__(self):
        print('Camera is open.')
        return self.vid

    def __exit__(self, *args):
        self.vid.release()
        cv2.destroyAllWindows()
        print('Camera is closed.')


def camera(fn: Callable[[cv2.Mat, Tuple[bool, list[float]]], None]):
    with Camera() as cap, hands.Hands(max_num_hands=1) as hand:
        try:
            while cap.isOpened():
                success: bool
                image: cv2.Mat

                success, image = cap.read()

                if not success:
                    raise BreakException(
                        "Can't receive frame (stream end?). Exiting ...")

                image = cv2.flip(image, 1)

                rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hand.process(rgb_img)

                has_hand = False
                distances = []
                if result.multi_hand_landmarks:  # type: ignore
                    has_hand = True
                    hand_landmarks = result.multi_hand_landmarks[0] # type: ignore

                    points = [(int(landmark.x*image.shape[1]), int(landmark.y*image.shape[0]))
                              for landmark in hand_landmarks.landmark]

                    (x, y), r = cv2.minEnclosingCircle(np.array(points))

                    wrist = hand_landmarks.landmark[0]
                    distances = [sqrt((wrist.x-other.x)**2+(wrist.y-other.y)**2+(wrist.z-other.z)**2)
                                 for other in hand_landmarks.landmark[1:]]

                    x_min, y_min = int(x - r), int(y - r)
                    x_max, y_max = int(x + r), int(y + r)

                    image = cv2.rectangle(
                        image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                fn(image, (has_hand, distances))

        except BreakException as e:
            print(e)
