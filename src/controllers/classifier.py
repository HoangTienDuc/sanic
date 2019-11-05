import numpy as np
import cv2
import logging

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc


def read_tensor(img,
                input_height=224,
                input_width=224,
                input_mean=0,
                input_std=255):
    float_caster = img.astype(np.float32)
    resized = cv2.resize(float_caster, (input_height, input_width))
    dims_expander = np.expand_dims(resized, axis=0)
    normalized = (dims_expander - input_mean) / input_std

    return normalized


def classify(tf_server, img, labels=['old_back', 'new_back', 'old', 'new']):
# def classify(tf_server, img, labels=['new', 'new_back', 'old', 'old_back']):
    """Predict the class of img by querying Serving server
    """

    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    # All-fields request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'cmt-classifier'
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img))

    # Do inference
    result_future = stub.Predict.future(request, None)

    # Extract output
    res = result_future.result().outputs['prediction']

    card_type = np.array(res.float_val).argsort()[-1]
    card_type2 = np.array(res.float_val).argsort()[-2]
   
    return labels[card_type], labels[card_type2]


def classify_gray(tf_server, img, labels=['old_back', 'new_back', 'old', 'new']):
    """
    Predict the class of img by querying Serving server
    """

    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    # All-fields request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'gray-classifier'
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img))

    # Do inference
    result_future = stub.Predict.future(request, None)

    # Extract output
    res = result_future.result().outputs['prediction']

    card_type = np.array(res.float_val).argsort()[-1]
    card_type2 = np.array(res.float_val).argsort()[-2]

    return labels[card_type], labels[card_type2]


def classify_gray_vs_rgb(tf_server, img, labels=['gray', 'rgb']):
    """
    Predict the class of img by querying Serving server
    """
    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    # All-fields request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'gray-vs-rgb-classifier'
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img))

    # Do inference
    result_future = stub.Predict.future(request, None)

    # Extract output
    res = result_future.result().outputs['prediction']

    card_type = np.array(res.float_val).argsort()[-1]
    return labels[card_type]


def classify_dl(tf_server, img, labels=['dl new', 'dl old', 'id new', 'id old']):
    """
    Predict the class of img by querying Serving server
    """
    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    # All-fields request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'blx-classifier'
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    request.inputs['image'].CopyFrom(
        tf.contrib.util.make_tensor_proto(img))

    # Do inference
    result_future = stub.Predict.future(request, None)

    # Extract output
    res = result_future.result().outputs['prediction']

    card_type = np.array(res.float_val).argsort()[-1]

    return labels[card_type]

