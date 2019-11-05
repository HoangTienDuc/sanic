import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc

from dh_segment import post_processing


CLASS = {
    # 'background': (0, 0, 0),
    'front_old': (0, 255, 0),
    'front_new': (255, 0, 0),
    'back_old': (255, 255, 0),
    'back_new': (0, 255, 255),
}


def mask_class_value(labels, val):
    # Labels contain classes
    # 0: background
    # 1: front old
    # 2: front new
    # 3: back old
    # 4: back new
    return np.where(labels != val, 0, labels)


def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_points_transform(img, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warped, rect.astype(np.int32)


if __name__ == '__main__':
    # setup
    MAX_MESSAGE_LENGTH = -1
    channel = grpc.insecure_channel('localhost:8501', options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = PredictionServiceStub(channel)
    request_detector = predict_pb2.PredictRequest()
    request_detector.model_spec.name = 'id'
    request_detector.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # ./tests is mounted to /tests
    for img_name in os.listdir('tests/id'):
        img = cv2.imread(os.path.join('tests/id', img_name))

        if img is not None:
            print(img_name)
            request_detector.inputs['filename'].CopyFrom(
                tf.contrib.util.make_tensor_proto(os.path.join('tests/id', img_name)))

            prediction_outputs = stub.Predict.future(request_detector, None)

            labels = tf.make_ndarray(prediction_outputs.result().outputs['labels'])[0]

            quad_img = img.copy()
            for i, (cl, color) in enumerate(CLASS.items()):
                label = mask_class_value(labels, i)
                mask = post_processing.thresholding(label)

                # re upscale
                orig_h, orig_w = tf.make_ndarray(prediction_outputs.result().outputs['original_shape'])
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # Detect quadrangles
                polygons = post_processing.find_boxes(mask, min_area=0.1)
                print('Detect {} {} quad boxes'.format(len(polygons), cl))

                os.makedirs('results/id/extracted', exist_ok=True)
                for i, polygon in enumerate(polygons):
                    quad, pts = four_points_transform(img, polygon)
                    cv2.polylines(quad_img, [pts.reshape((-1,1,2))], True, color, thickness=5)

            cv2.imwrite(os.path.join('results/id', img_name), quad_img)
            # Save perspective transformed image
            # cv2.imwrite('results/id/extracted/{}_{}'.format(i, img_name), quad)
