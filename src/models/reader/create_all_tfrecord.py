import argparse
import glob
import sys
import os
import math
import unicodedata

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from tqdm import tqdm

from dataset import get_config
import utils


## Get ENV config
PREFIX = 'READER_'
TRANSFORM = os.getenv(PREFIX + 'TRANSFORM') or 'pad'


def encode_utf8_string(text, length, charset):
    null_char_id = len(charset) - 1

    char_ids_padded = [null_char_id] * length
    char_ids_unpadded = [null_char_id] * len(text)
    for i, c in enumerate(list(text)):
        hash_id = charset[c]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id

    return char_ids_padded, char_ids_unpadded


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required parameters
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--phase', type=str, required=True)

    # config parameters
    parser.add_argument('--charset-file', type=str, default='data/charset_size.txt')
    args = parser.parse_args()

    # Read charset dictionary
    charset = {}
    with open(args.charset_file) as dict_file:
        for line in dict_file:
            k, v = line[:-1].split('\t')
            charset[v] = int(k)

    # Read dataset config
    config = get_config('all')

    tfrecord_writer = tf.python_io.TFRecordWriter(args.output)
    for type_dir in glob.glob(os.path.join(args.data_dir, '*')):
        for class_dir in glob.glob(os.path.join(type_dir, '*')):
            print('>> Reading', class_dir)

            # Read phase label
            img2label = {}
            lines = open(os.path.join(class_dir, args.phase+'.txt')).readlines()
            for l in lines:
                try:
                    img_name, label = l.strip().split('\t')
                except ValueError as e:
                    print("ERROR line:", l)
                    continue

                img2label[img_name] = label.strip()

            # Read image
            for img_path in tqdm(glob.glob(os.path.join(class_dir, 'jpg/*'))):
                img_name = img_path.split('/')[-1]

                try:
                    img = Image.open(img_path)
                except OSError:
                    print("ERROR image:", img_path)

                height, width, channel = config['image_shape']

                if TRANSFORM == 'pad':
                    new_img = utils.pad_image_keep_ratio(img, width, height)
                elif TRANSFORM == 'resize':
                    new_img = utils.resize_image(img, width, height)

                if new_img == None:
                    continue

                if channel == 1:
                    # Grayscale
                    new_img = new_img.convert('L')

                # Get size
                new_w, new_h = new_img.size

                try:
                    assert new_w == width
                except AssertionError:
                    print('new width: {}, expected width: {}'.format(new_w, width))

                # Maybe img_name not in images_dir
                if img_name not in img2label:
                    continue

                text = img2label[img_name]
                # Force uppercase
                text = text.upper()
                # Normalize to NFC
                text = unicodedata.normalize('NFC', text)
                # Strange label
                if len(text) > int(config['max_sequence_length']):
                    continue

                char_ids_padded, char_ids_unpadded = encode_utf8_string(
                    text=text,
                    charset=charset,
                    length=config['max_sequence_length'])

                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image/encoded': _bytes_feature(new_img.tobytes()),
                        'image/format': _bytes_feature(b"raw"),
                        'image/width': _int64_feature([new_w]),
                        'image/orig_width': _int64_feature([new_w]),
                        'image/class': _int64_feature(char_ids_padded),
                        'image/unpadded_class': _int64_feature(char_ids_unpadded),
                        'image/text': _bytes_feature(bytes(text, 'utf-8')),
                        }
                ))
                tfrecord_writer.write(example.SerializeToString())

    tfrecord_writer.close()
    sys.stdout.flush()
