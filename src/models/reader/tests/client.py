import argparse
import threading
import time
import sys
import os
sys.path.append('..')

from PIL import Image
import numpy as np
import tensorflow as tf
import grpc

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from dataset import get_config
import utils


class _ResultCounter(object):
    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(result_counter):
    def _callback(result_future):
        """Callback function.
        Calculates the statistics for the prediction result.
        Args:
          result_future: Result future of the RPC.
        """
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            words = result_future.result().outputs['output'].string_val
            words = [w.decode('utf-8').strip(u'\u2591') for w in words]
            print(words)
            sys.stdout.flush()

        result_counter.inc_done()
        result_counter.dec_active()

    return _callback


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--reader_name', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=10)
    args = parser.parse_args()

    # Connect to server
    channel = grpc.insecure_channel('localhost:8500')
    stub = PredictionServiceStub(channel)

    # Prepare request object
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.reader_name
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Prepare input
    conf = get_config('all')

    h, w, _ = conf['image_shape']
    result_counter = _ResultCounter(len(os.listdir(args.images_dir)), args.concurrency)
    tic = time.time()
    for img_name in os.listdir(args.images_dir):
        img_path = os.path.join(args.images_dir, img_name)

        img = Image.open(img_path)
        img = utils.pad_image_keep_ratio(img, w, h)
        img = np.array(img)

        # prepare batch input
        inputs = np.zeros((args.batch_size, *img.shape), dtype=np.uint8)
        inputs[0, ...] = img

        request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(inputs))

        result_counter.throttle()

        # Do inference with OCR server
        try:
            result_future = stub.Predict.future(request, None)
        except Exception as e:
            logging.error('OCR service error:', e)
            sys.exit(1)

        result_future.add_done_callback(
            _create_rpc_callback(result_counter))

    toc = time.time()

    print(result_counter.get_error_rate())
    print('Time: {}s'.format(toc - tic))


if __name__ == '__main__':
  tf.app.run()
