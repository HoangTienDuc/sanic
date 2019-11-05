import os
import datetime
import numpy as np
import cv2

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc

from src.models.dhSegment.dh_segment import post_processing


CLASS = {
    # 'background': (0, 0, 0),
    'back_old': (0, 255, 0),
    'front_old': (255, 0, 0),
    'front_new': (255, 255, 0),
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


def segment(tf_server, image, image_name):
    # Connect to server
    MAX_MESSAGE_LENGTH = -1
    channel = grpc.insecure_channel(tf_server, options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = PredictionServiceStub(channel)

    # Prepare request object
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'segmentator-id'
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Copy image's name into request's content
    request.inputs['filename'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_name))

    # Do inference
    prediction_outputs = stub.Predict.future(request, None)

    # Extract output
    labels = tf.make_ndarray(prediction_outputs.result().outputs['labels'])[0]

    quad_img = image.copy()
    polygons = list()
    for i, (cl, color) in enumerate(CLASS.items()):
        label = mask_class_value(labels, i)
        mask = post_processing.thresholding(label)

        # re upscale
        orig_h, orig_w = tf.make_ndarray(prediction_outputs.result().outputs['original_shape'])
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Detect quadrangles
        p = post_processing.find_boxes(mask, min_area=0.1)
        if len(p) > 0:
            polygons.append(p)
        
    #     print('Detect {} {} quad boxes'.format(len(polygons), cl))
    # print(polygons)
    # print(polygons[0])
    # print(polygons[0][0])
    # print(polygons[0][0][0])
    # print(polygons[0][0][1])
    if len(polygons) > 0:
        polygons = sorted(polygons, key=lambda x: x[0][1], reverse=True)
        # for i, polygon in enumerate(polygons):
        # print(polygons[0])
        quad, pts = four_points_transform(image, polygons[0][0][0])
        # print(pts)
        bottom_right = (pts[np.argmax(np.array(pts), axis=0)][0][0] + 20, pts[np.argmax(np.array(pts), axis=0)][1][1] + 20)
        top_left = (pts[np.argmin(np.array(pts), axis=0)][0][0] - 20, pts[np.argmin(np.array(pts), axis=0)][1][1] - 20)
        # if top_left[0] < 0:
        #     top_left = (0, top_left[1])
        # if top_left[1] < 0:
        #     top_left[1] = (top_left[0], 0)
        # print(quad_img.shape)
        # if bottom_right[0] > quad_img.shape[1]:
        #     bottom_right = (quad_img.shape[1], bottom_right[1])
        # if bottom_right[1] > quad_img.shape[0]:
        #     bottom_right = (bottom_right[0], quad_img.shape[0])
        # # cv2.polylines(quad_img, [pts.reshape((-1,1,2))], True, color, thickness=5)
        # cv2.circle(quad_img, top_left, 5, (255,0,0))
        # cv2.circle(quad_img, bottom_right, 5, (0,255,0))
        # print(top_left)
        # print(bottom_right)

        cv2.imwrite('segment.jpg', quad_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]])
        return quad_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    else:
        return None
