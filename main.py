import os
import cv2
import dlib
import argparse
import numpy as np

from wide_resnet import WideResNet
from contextlib import contextmanager
from keras.utils.data_utils import get_file

models = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.18-4.06.hdf5"
modhash = '89f56a39a78454e96379348bddd78c0d'


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


# @contextmanager
# def video_capture(*args, **kwargs):
#     cap = cv2.VideoCapture(*args, **kwargs)
#     try:
#         yield cap
#     finally:
#         cap.release()


# def yield_images():
#     # capture video
#     with video_capture(0) as cap:
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#         while True:
#             # get video frame
#             ret, img = cap.read()
#
#             if not ret:
#                 raise RuntimeError("Failed to capture image")
#
#             yield img


def main():
    args = get_args()
    depth = args.depth
    k = args.width
    weight_file = args.weight_file

    if not weight_file:
        weight_file = get_file("weights.18-4.06.hdf5", models, cache_subdir="models",
                               file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))

    # for face detection
    detector = dlib.get_frontal_face_detector()

    # load model and weights
    img_size = 64
    model = WideResNet(img_size, depth=depth, k=k)()
    model.load_weights(weight_file)


        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_weight, _ = np.shape(input_img)

        # detect faces using dlib detector
        detected = detector(input_img, 1)
        faces = np.empty((len(detected), img_size, img_size, 3))

        if len(detected) > 0:
            for i, detection in enumerate(detected):
                x1, y1, x2, y2, weight, height = detection.left(), detection.top(), detection.right() + 1, detection.bottom() + 1, detection.width(), detection.height()
                xw1 = max(int(x1 - 0.4 * weight), 0)
                yw1 = max(int(y1 - 0.4 * height), 0)
                xw2 = min(int(x2 + 0.4 * weight), img_weight - 1)
                yw2 = min(int(y2 + 0.4 * height), img_height - 1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                # result = [xw1, yw1, xw2, yw2]
                # print(result)

            # predict ages and genders of the detected faces
            results = model.predict(faces)
            predicted_genders = results[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = results[1].dot(ages).flatten()

            # draw results
            for i, detection in enumerate(detected):
                label = "{}, {}".format(int(predicted_ages[i]),
                                        "F" if predicted_genders[i][0] > 0.5 else "M")
                draw_label(img, (detection.left(), detection.top()), label)

        cv2.imshow("result", img)
        key = cv2.waitKey(30)
        if key == 27:
            break
