import os
import datetime

import tensorflow as tf
import cv2
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc

from src.common import CROPPED_DIR


def postprocess(output_dict):
    """Postprocessing of output_dict, to be put in controller"""
    batch_size = len(output_dict['num_detections'])

    output_dict['num_detections'] = [
        int(output_dict['num_detections'][i]) for i in range(batch_size)]
    output_dict['detection_classes'] = [output_dict[
        'detection_classes'][i].astype(np.uint8) for i in range(batch_size)]
    output_dict['detection_boxes'] = [output_dict['detection_boxes'][i]
                                      for i in range(batch_size)]
    output_dict['detection_scores'] = [
        output_dict['detection_scores'][i] for i in range(batch_size)]
    all_corners = []
    for i in range(batch_size):
        corner_0 = []
        corner_1 = []
        corner_2 = []
        corner_3 = []

        for j in range(len(output_dict['detection_classes'][0])):
            if output_dict['detection_classes'][0][j] == 1:
                corner_0.append({'score': output_dict['detection_scores'][0][j],
                                 'box': output_dict['detection_boxes'][0][j]})
            if output_dict['detection_classes'][0][j] == 2:
                corner_1.append({'score': output_dict['detection_scores'][0][j],
                                 'box': output_dict['detection_boxes'][0][j]})
            if output_dict['detection_classes'][0][j] == 3:
                corner_2.append({'score': output_dict['detection_scores'][0][j],
                                 'box': output_dict['detection_boxes'][0][j]})
            if output_dict['detection_classes'][0][j] == 4:
                corner_3.append({'score': output_dict['detection_scores'][0][j],
                                 'box': output_dict['detection_boxes'][0][j]})

        corner_0_sorted = sorted(corner_0, key=lambda k: k['score'], reverse=True)
        corner_1_sorted = sorted(corner_1, key=lambda k: k['score'], reverse=True)
        corner_2_sorted = sorted(corner_2, key=lambda k: k['score'], reverse=True)
        corner_3_sorted = sorted(corner_3, key=lambda k: k['score'], reverse=True)

        if len(corner_0_sorted) != 0 and len(corner_1_sorted) != 0 and \
                len(corner_2_sorted) != 0 and len(corner_3_sorted) != 0:
            all_corners = [corner_0_sorted[0], corner_1_sorted[0], corner_2_sorted[0], corner_3_sorted[0]]

    return all_corners


def transform(image, corners, padding=5):
    """ Visualization """
    points = []
    height, width, _ = image.shape

    for corner in corners:
        box = corner['box']
        if box is None:
            return
        points.append(((box[1] + box[3]) / 2 * width, (box[0] + box[2]) / 2 * height))

    marker = np.array(((0 + padding, 0 + padding), (800 - padding, 0 + padding), (0 + padding, 500 - padding),
                       (800 - padding, 500 - padding)), dtype=np.float32)
    h = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), marker)
    cropped = cv2.warpPerspective(image, h, (800, 500))

    # DEBUG - visualization
    # cv2.circle(image, tuple(map(int, points[0])), 10, (0, 255, 0), -1)
    # cv2.circle(image, tuple(map(int, points[1])), 10, (0, 0, 255), -1)
    # cv2.circle(image, tuple(map(int, points[2])), 10, (255, 0, 0), -1)
    # cv2.circle(image, tuple(map(int, points[3])), 10, (0, 255, 255), -1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.rectangle(image, (int(corners[0]['box'][1]*width), int(corners[0]['box'][0]*height)),
    #               (int(corners[0]['box'][3]*width), int(corners[0]['box'][2]*height)), (0, 255, 0), 2)
    # cv2.rectangle(image, (int(corners[1]['box'][1] * width), int(corners[1]['box'][0] * height)),
    #               (int(corners[1]['box'][3] * width), int(corners[1]['box'][2] * height)), (0, 0, 255), 2)
    # cv2.rectangle(image, (int(corners[2]['box'][1] * width), int(corners[2]['box'][0] * height)),
    #               (int(corners[2]['box'][3] * width), int(corners[2]['box'][2] * height)), (255, 0, 0), 2)
    # cv2.rectangle(image, (int(corners[3]['box'][1] * width), int(corners[3]['box'][0] * height)),
    #               (int(corners[3]['box'][3] * width), int(corners[3]['box'][2] * height)), (0, 255, 255), 2)
    # cv2.imshow("abc", image)
    # # wait for keypress; capture it
    # k = cv2.waitKey(0)
    # if k == 27:  # this should be ESC
    #     return  # e.g. end the program

    return cropped, points


def crop(tf_server, image, image_name, card_type, save_debug_images):
    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    # Cropper request
    request_cropper = predict_pb2.PredictRequest()
    if card_type == 'old' or card_type == 'old_gray':
        pad = 10
        request_cropper.model_spec.name = 'cropper-detector'
    elif card_type == 'new' or card_type == 'new_gray':
        pad = 35
        request_cropper.model_spec.name = 'cropper-new-detector'
    elif card_type == 'old_back':
        pad = 10
        request_cropper.model_spec.name = 'cropper-back-detector'
    elif card_type == 'new_back':
        pad = 30
        request_cropper.model_spec.name = 'cropper-new-back-detector'
    
    request_cropper.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Pass image to the request
    image_raw_np_expanded = np.expand_dims(image, axis=0)
    request_cropper.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_raw_np_expanded))

    result_cropper = stub.Predict.future(request_cropper, None)

    # Create result dictionary
    crop_dict = dict()
    crop_dict['num_detections'] = tf.make_ndarray(result_cropper.result().outputs['num_detections'])
    crop_dict['detection_boxes'] = tf.make_ndarray(result_cropper.result().outputs['detection_boxes'])
    crop_dict['detection_classes'] = tf.make_ndarray(result_cropper.result().outputs['detection_classes'])
    crop_dict['detection_scores'] = tf.make_ndarray(result_cropper.result().outputs['detection_scores'])

    corners = postprocess(crop_dict)
    cropped_path = []

    if len(corners) < 4:
        return None, None, None
    else:
        # Process each image in the batch
        # for corner in corners:
        img_cropped, points = transform(image, corners, padding=pad)

        if save_debug_images == '1':
            # Check if result path exits
            path_cropped = os.path.join(CROPPED_DIR, datetime.date.today().isoformat())
            if not os.path.exists(path_cropped):
                os.makedirs(path_cropped)

            cv2.imwrite(os.path.join(path_cropped, image_name), img_cropped)
            cropped_path.append(os.path.join(path_cropped, image_name))

        return img_cropped, cropped_path, points


def transform_fallback(image, corners, card_type, padding=5):
    """ cnt. """
    points = []
    height, width, _ = image.shape

    if card_type == 'old' or card_type == 'old_back' or card_type == 'new_back' or \
                                card_type == 'old_gray' or card_type == 'new_gray':
        for corner in corners:
            box = corner['box']
            if box is None:
                return
            points.append(((box[1] + box[3]) / 2 * width, (box[0] + box[2]) / 2 * height))

    elif card_type == 'new':
        box_0 = corners[0]['box']
        box_1 = corners[1]['box']
        box_2 = corners[2]['box']
        box_3 = corners[3]['box']

        # Normal orientation
        if box_1[1] * width > box_0[1] * width + 50:
            points.append((box_0[1] * width, box_0[0] * height))
            points.append((((box_1[3] - box_1[1]) * 3 / 4 + box_1[1]) * width,
                           ((box_1[2] - box_1[0]) * 1 / 4 + box_1[0]) * height))

            points.append((box_2[1] * width, box_2[2] * height))
            points.append(((box_3[1] + box_3[3]) / 2 * width, (box_3[0] + box_3[2]) / 2 * height))
        # 90 degree clockwise
        else:
            if box_0[0] * height < box_1[0] * height + 50:
                if box_0[0] > box_3[0] and box_0[1] > box_3[1]:
                    points.append((box_0[3] * width, box_0[2] * height))
                    points.append((((box_1[3] - box_1[1]) * 1 / 4 + box_1[1]) * width,
                                        ((box_1[2] - box_1[0]) * 3 / 4 + box_1[0]) * height))
                    points.append((box_2[3] * width, box_2[0] * height))
                    points.append(((box_3[1] + box_3[3]) / 2 * width, (box_3[0] + box_3[2]) / 2 * height))
                else:
                    points.append((box_0[3] * width, box_0[0] * height))
                    points.append(((box_1[3] - (box_1[3] - box_1[1]) / 4) * width,
                                        (box_1[2] - (box_1[2] - box_1[0]) / 4) * height))
                    points.append((box_2[1] * width, box_2[0] * height))
                    points.append(((box_3[1] + box_3[3]) / 2 * width, (box_3[0] + box_3[2]) / 2 * height))
            else:
                # 90 Degree Counter_clockwise
                points.append((box_0[1] * width, box_0[2] * height))
                points.append(((box_1[1] + (box_1[3] - box_1[1]) / 4) * width,
                               (box_1[0] + (box_1[2] - box_1[0]) / 4) * height))
                points.append((box_2[3] * width, box_2[2] * height))
                points.append(((box_3[1] + box_3[3]) / 2 * width, (box_3[0] + box_3[2]) / 2 * height))

    marker = np.array(((0 + padding, 0 + padding), (800 - padding, 0 + padding), (0 + padding, 500 - padding),
                       (800 - padding, 500 - padding)), dtype=np.float32)
    h = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), marker)
    cropped = cv2.warpPerspective(image, h, (800, 500))

    # DEBUG - visualization
    # cv2.circle(image, tuple(map(int, points[0])), 10, (0, 255, 0), -1)
    # cv2.circle(image, tuple(map(int, points[1])), 10, (0, 0, 255), -1)
    # cv2.circle(image, tuple(map(int, points[2])), 10, (255, 0, 0), -1)
    # cv2.circle(image, tuple(map(int, points[3])), 10, (0, 255, 255), -1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.rectangle(image, (int(corners[0]['box'][1]*width), int(corners[0]['box'][0]*height)),
    #               (int(corners[0]['box'][3]*width), int(corners[0]['box'][2]*height)), (0, 255, 0), 2)
    # cv2.rectangle(image, (int(corners[1]['box'][1] * width), int(corners[1]['box'][0] * height)),
    #               (int(corners[1]['box'][3] * width), int(corners[1]['box'][2] * height)), (0, 0, 255), 2)
    # cv2.rectangle(image, (int(corners[2]['box'][1] * width), int(corners[2]['box'][0] * height)),
    #               (int(corners[2]['box'][3] * width), int(corners[2]['box'][2] * height)), (255, 0, 0), 2)
    # cv2.rectangle(image, (int(corners[3]['box'][1] * width), int(corners[3]['box'][0] * height)),
    #               (int(corners[3]['box'][3] * width), int(corners[3]['box'][2] * height)), (0, 255, 255), 2)
    # cv2.imshow("abc", image)
    # # wait for keypress; capture it
    # k = cv2.waitKey(0)
    # if k == 27:  # this should be ESC
    #     return  # e.g. end the program

    return cropped, points


# Cropper khong xoay cheo
def crop_fallback(tf_server, image, image_name, card_type, save_debug_images):
    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    # Cropper request
    request_cropper = predict_pb2.PredictRequest()
    if card_type == 'old':
        pad = 20
        request_cropper.model_spec.name = 'cropper-detector-fallback'
    elif card_type == 'new':
        pad = 20
        request_cropper.model_spec.name = 'cropper-new-detector-fallback'
    elif card_type == 'old_back':
        pad = 10
        request_cropper.model_spec.name = 'cropper-back-detector-fallback'
    elif card_type == 'new_back':
        pad = 30
        request_cropper.model_spec.name = 'cropper-new-back-detector-fallback'
    elif card_type == 'old_gray':
        pad = 15
        request_cropper.model_spec.name = 'cropper-gray-detector-fallback'
    elif card_type == 'new_gray':
        pad = 15
        request_cropper.model_spec.name = 'cropper-new-gray-detector-fallback'
    request_cropper.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Pass image to the request
    image_raw_np_expanded = np.expand_dims(image, axis=0)
    request_cropper.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image_raw_np_expanded))

    result_cropper = stub.Predict.future(request_cropper, None)

    # Create result dictionary
    crop_dict = dict()
    crop_dict['num_detections'] = tf.make_ndarray(result_cropper.result().outputs['num_detections'])
    crop_dict['detection_boxes'] = tf.make_ndarray(result_cropper.result().outputs['detection_boxes'])
    crop_dict['detection_classes'] = tf.make_ndarray(result_cropper.result().outputs['detection_classes'])
    crop_dict['detection_scores'] = tf.make_ndarray(result_cropper.result().outputs['detection_scores'])

    corners = postprocess(crop_dict)
    cropped_path = []

    if len(corners) < 4:
        return None, None, None
    else:
        # Process each image in the batch
        # for corner in corners:
        img_cropped, points = transform_fallback(image, corners, card_type=card_type, padding=pad)

        if save_debug_images == '1':
            # Check if result path exits
            path_cropped = os.path.join(CROPPED_DIR, datetime.date.today().isoformat())
            if not os.path.exists(path_cropped):
                os.makedirs(path_cropped)

            cv2.imwrite(os.path.join(path_cropped, image_name), cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
            cropped_path.append(os.path.join(path_cropped, image_name))

        return img_cropped, cropped_path, points

