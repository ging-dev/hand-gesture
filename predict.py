import cv2
import numpy as np
import tensorflow as tf
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
from lib import camera, BreakException
from typing import Tuple


def predict_program(image: cv2.Mat, hand_detect: Tuple[bool, list[float]]):
    global model, le
    assert(isinstance(model, Sequential))

    has_hand, distances = hand_detect
    if has_hand:
        features = np.array(distances).reshape(1, 20)
        predict = np.argmax(model.predict(features, verbose='0')).reshape(1)
        action = le.inverse_transform(predict)
        cv2.putText(image, f'{action[0]}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("collect", image)

    match cv2.waitKey(5) & 0xFF:
        case 27:
            raise BreakException("Shutdown...")

if __name__ == '__main__':
    model = tf.keras.models.load_model('./gesture.h5')
    le = LabelEncoder()
    le.fit_transform(['rock', 'paper', 'scissor', 'like', 'ok'])
    camera(predict_program)
