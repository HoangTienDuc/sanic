import os
import sys
sys.path.append('..')

import numpy as np
from PIL import Image

from tqdm import tqdm
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session

import common_flags
import data_provider
from dataset import get_config
import utils as reader_utils


FLAGS = flags.FLAGS
common_flags.define()

flags.DEFINE_string('images_dir', '',
                    'Directory contains images')


def load_images(image_dir, batch_size):
  height, width, channel = get_config('all')['image_shape']
  images_actual_data = np.ndarray(shape=(batch_size, height, width, channel),
                                  dtype='uint8')
  for i in tqdm(range(batch_size)):
    path = os.path.join(image_dir, os.listdir(image_dir)[i])
    print("Reading %s" % path)

    id_box = Image.open(path)
    id_box = reader_utils.pad_image_keep_ratio(id_box, width, height)
    id_box = np.array(id_box)
    images_actual_data[i, ...] = id_box

  return images_actual_data


def create_model(batch_size):
  height, width, channel = get_config('all')['image_shape']
  dataset = common_flags.create_dataset(class_name='all', split_name=FLAGS.split_name)

  model = common_flags.create_model(
    num_char_classes=dataset.num_char_classes,
    seq_length=dataset.max_sequence_length,
    num_views=dataset.num_of_views,
    null_code=dataset.null_code,
    charset=dataset.charset)

  raw_images = tf.placeholder(tf.uint8, shape=[batch_size, height, width, channel])
  images = tf.map_fn(data_provider.preprocess_image, raw_images, dtype=tf.float32)
  endpoints = model.create_base(images, labels_one_hot=None)
  return raw_images, endpoints


def run(checkpoint, batch_size, image_dir):
  images_placeholder, endpoints = create_model(batch_size)
  images_data = load_images(image_dir, batch_size)
  session_creator = monitored_session.ChiefSessionCreator(
    checkpoint_filename_with_path=checkpoint)
  with monitored_session.MonitoredSession(
      session_creator=session_creator) as sess:
    predictions = sess.run(endpoints.predicted_text,
                           feed_dict={images_placeholder: images_data})
  return predictions.tolist()


def main(_):
  predictions = run(FLAGS.checkpoint, FLAGS.batch_size, FLAGS.images_dir)
  for line in predictions:
    print(line.decode('utf-8'))


if __name__ == '__main__':
  tf.app.run()
