"""Prepare data for region classification, save as data.csv
id, x1 (int), y1, x2, y2, class (str)
"""
"""Improve name detector with both accuracy and inference time
"""

import argparse
import sys
import os

import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm

from grpc.beta import implementations
from tensorflow_serving.apis import prediction_service_pb2

# Make this script able to call lib module from parent level
sys.path.append(os.getcwd())
LINE_DETECTOR_DIR = '../../line_detector/'
sys.path.append(LINE_DETECTOR_DIR)

from lib.fast_rcnn.config import cfg, cfg_from_file
import detector


ID_REGION = [(380, 100), (800, 180)]
BIRTHDAY_REGION = [(380, 250), (800, 320)]
FIRST_NAME_REGION = [(330,150), (800, 250)]
SECOND_NAME_REGION = [(200, 210), (800, 285)]
FIRST_ADDRESS_REGION = [(480, 370), (800, 460)]
SECOND_ADDRESS_REGION = [(0, 420), (800, 500)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--serving', type=str, default='localhost:9000')
    parser.add_argument('--out', type=str, required=True,
                        help='Path to output CSV file')
    args = parser.parse_args()
    
    # Config file
    cfg_from_file(os.path.join(LINE_DETECTOR_DIR, 'text.yml'))
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)

    # Load pb graph
    with tf.gfile.FastGFile(os.path.join(LINE_DETECTOR_DIR, 'models/ctpn.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    sess.run(tf.global_variables_initializer())

    # Connect to Reader server
    host, port = args.serving.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    with open(args.out, 'w') as f:
        f.write('id,x1,y1,x2,y2,class\n')
        for img_name in tqdm(os.listdir(args.data_dir)):
            img_path = os.path.join(args.data_dir, img_name)
            img_id = img_name.split('.')[0]
            img = cv2.imread(img_path)

            # Detect lines
            boxes = detector.query_pb(sess, cfg, img)

            results = [
                ('id', detector.extract_info(stub, img, boxes, ID_REGION, margin=0)),
                ('name1', detector.extract_info(stub, img,
                                                boxes, FIRST_NAME_REGION, margin=0)),
                ('name2', detector.extract_info(stub, img,
                                                boxes, SECOND_NAME_REGION, margin=0)),
                ('birthday', detector.extract_info(stub, img, boxes, BIRTHDAY_REGION, 
                                              h_threshold=20, margin=0)),
                ('address1', detector.extract_info(stub, img,
                                                   boxes, FIRST_ADDRESS_REGION, margin=0)),
                ('address2', detector.extract_info(stub, img,
                                                   boxes, SECOND_ADDRESS_REGION, margin=0))
            ]

            labels = [None] * len(boxes)
            for (label, coord) in results:
                if coord is not None:
                    for i, box in enumerate(boxes):
                        if coord[-1] == box[-1]:
                            labels[i] = label

            for (label, box) in zip(labels, boxes):
                (x1, y1), (x2, y2) = box
                f.write('{},{},{},{},{},{}\n'.format(img_id, x1, y1, x2, y2, label))
