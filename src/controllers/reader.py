import os
import logging
import functools
import traceback

import cv2
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc

from src.models.reader.dataset import get_config
import src.models.reader.utils as reader_utils

from src.common import DEBUG_IMAGE


def read_batch(tf_server, reader_name, img, filename, box_coords, save_dir, batch_size=32):
    """Do inference with Reader OCR server, accept 4D input

    Args:
        tf_server: Name of serving host (e.g. localhost:8500)
        reader_name: the target reader to do inference (e.g. id, birthday, name)
        img: Numpy array of image
        filename: Name of the file to write into disk
        box_coords: bounding box coordinates [[(xmin, ymin), (xmax, ymax)], ...]
        save_dir: folder to contain saved files

    Returns:
        [(word1, prob1), ...]
    """
    if len(box_coords) == 0 or all(x is None for x in box_coords):
        return []

    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    request = predict_pb2.PredictRequest()
    request.model_spec.name = reader_name
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Prepare input
    class_name = 'all'
    conf = get_config(class_name)
    if conf == None:
        logging.error('Wrong config type')
        return None

    h, w, c = conf['image_shape']
    inputs = np.zeros(shape=(batch_size, h, w, c),
                      dtype='uint8')

    if len(box_coords) < 32:
        for i, ((xmin, ymin), (xmax, ymax)) in enumerate(box_coords):
            try:
                box = img[ymin:ymax, xmin:xmax]
                # Padding
                box = Image.fromarray(box.astype('uint8'), 'RGB')
                box = reader_utils.pad_image_keep_ratio(box, w, h)
                box = np.array(box)

                # Save debug image
                if DEBUG_IMAGE:
                    filepath = os.path.join(save_dir, '{}_{}_{}'.format(reader_name, i, filename))

                    # TODO: put into external queue (e.g. redis) to save time
                    cv2.imwrite(filepath, box)

                # Prepare input
                inputs[i, ...] = box
            except:
                traceback.print_exc()
                continue
    else:
        return []

    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(inputs))

    # Do inference with OCR server
    try:
        result_future = stub.Predict.future(request, None)
    except:
        logging.error(traceback.format_exc())
        return None

    # Extract output
    words = result_future.result().outputs['output'].string_val
    # Decode by UTF-8
    # Strip null character
    words = [word.decode('utf-8').strip(u'\u2591') for word in words[:len(box_coords)]]
    words = list(filter(lambda w: len(w) > 0, words))

    # Calculate probability
    probs = result_future.result().outputs['prob'].float_val
    probs = np.array(probs).reshape(batch_size, -1)
    probs_old_style = [functools.reduce(lambda x, y: x * y, prob[:len(words[i])]) for i, prob in enumerate(probs[:len(words)])]
    probs = [prob[-1] for i, prob in enumerate(probs[:len(words)])]

    return list(zip(words, probs, probs_old_style))


def read(tf_server, reader_name, img, filename, box_coord, save_dir):
    """Do inference with Reader OCR server, accept 4D input

    Args:
        tf_server: Name of serving host (e.g. localhost:8500)
        reader_name: the target reader to do inference (e.g. id, birthday, name)
        img: Numpy array of image
        filename: Name of the file to write into disk
        box_coord: bounding box coordinate (x1,y1)
        save_dir: folder to contain saved files

    Returns:
        (word, prob)
    """
    if box_coord is None:
        return {
            'result': ''
        }


    # Save crop box into disk
    (xmin, ymin), (xmax, ymax) = box_coord

    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    request = predict_pb2.PredictRequest()
    request.model_spec.name = reader_name
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Prepare input
    class_name = 'all'
    conf = get_config(class_name)
    if conf == None:
        logging.error('Wrong config type')
        return None

    h, w, c = conf['image_shape']
    inputs = np.ndarray(shape=(1, h, w, c),
                        dtype='uint8')

    try:
        (xmin, ymin), (xmax, ymax) = box_coord
        box = img[ymin:ymax, xmin:xmax]

        # Save debug image
        if DEBUG_IMAGE:
            filepath = os.path.join(save_dir, '{}_{}'.format(reader_name, filename))

            # TODO: put into external queue (e.g. redis) to save time
            cv2.imwrite(filepath, box)

        # Padding
        box = Image.fromarray(box.astype('uint8'), 'RGB')
        box = reader_utils.pad_image_keep_ratio(box, w, h)
        box = np.array(box)

        # Prepare input
        inputs[0, ...] = box
    except:
        traceback.print_exc()
        return {
            'result': ''
        }

    request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(inputs))

    # Do inference with OCR server
    try:
        result_future = stub.Predict.future(request, None)
    except:
        logging.error(traceback.format_exc())
        return None

    # Extract output
    word = result_future.result().outputs['output'].string_val[0]
    # Decode by UTF-8
    # Strip null character
    word = word.decode('utf-8').strip(u'\u2591')

    # Calculate probability
    probs = result_future.result().outputs['prob'].float_val
    prob = probs[-1]
    prob_old_style = functools.reduce(lambda x, y: x * y, probs[:len(word)])

    return {
        'result': word,
        'prob': prob,
        'prob_old_style': prob_old_style,
    }


def read_tensor(img,
                input_height=96,
                input_width=96,
                input_mean=0,
                input_std=255):
    float_caster = img.astype(np.float32)
    resized = cv2.resize(float_caster, (input_height, input_width))
    dims_expander = np.expand_dims(resized, axis=0)
    normalized = (dims_expander - input_mean) / input_std

    return normalized


def convert_bbox_coordinates(image, box, use_normalized_coordinates=True):
    """Convert to non-normalized coordinates
    box = [ymin, xmin, ymax, xmax]"""
    coord = []
    im_height, im_width, _ = image.shape

    if use_normalized_coordinates:
        (left, right, top, bottom) = (int(box[1] * im_width), int(box[3] * im_width),
                                      int(box[0] * im_height), int(box[2] * im_height))
    else:
        (left, right, top, bottom) = (int(box[1]), int(box[3]), int(box[0]), int(box[2]))

    coord = [(left, top), (right, bottom)]
    return coord


def load_image_into_numpy_array(image):
    # (im_width, im_height) = image.size
    im_height, im_width, _ = image.shape
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
