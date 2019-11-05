import numpy as np

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
import grpc


def detect(tf_server, model_name, image):
    # Connect to server
    channel = grpc.insecure_channel(tf_server)
    stub = PredictionServiceStub(channel)

    # Prepare request object
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Copy image into request's content
    cropped_image_np_expanded = np.expand_dims(image, axis=0)
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(cropped_image_np_expanded))

    # Do inference
    result = stub.Predict.future(request, None)

    # Extract output
    # Create result dictionary
    detected_result = dict()
    detected_result['boxes'] = tf.make_ndarray(result.result().outputs['detection_boxes'])
    detected_result['classes'] = tf.make_ndarray(result.result().outputs['detection_classes'])
    detected_result['scores'] = tf.make_ndarray(result.result().outputs['detection_scores'])

    return detected_result
