import os
import sys
import re
import logging
import configparser

import tensorflow as tf
from tensorflow.contrib import slim


# Load config
config = configparser.ConfigParser()
conf_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
if not os.path.exists(conf_file):
  print('Config file {} not found'.format(conf_file))
  sys.exit(1)
config.read(conf_file)


def get_config(class_name):
  try:
    conf = config[class_name]
  except KeyError as e:
    logging.error('{} not in config.ini'.format(class_name))
    return None

  return {
    'name': conf['name'],
    'splits': {
      'train': { 'pattern': 'train.tfrecord' },
      'test': { 'pattern': 'test.tfrecord' },
      'real': { 'pattern': 'real.tfrecord' }
    },
    'image_shape': (int(conf['image_height']), int(conf['image_width']), int(conf['image_channel'])),
    'num_of_views': 1,
    'max_sequence_length': int(conf['max_sequence_length']),
    'null_code': 118
  }


def read_charset(filename, null_character=u'\u2591'):
  """Reads a charset definition from a tab separated text file.

  charset file has to have format compatible with the FSNS dataset.

  Args:
    filename: a path to the charset file.
    null_character: a unicode character used to replace '<null>' character. the
      default value is a light shade block '░'.

  Returns:
    a dictionary with keys equal to character codes and values - unicode
    characters.
  """
  pattern = re.compile(r'(\d+)\t(.+)')
  charset = {}
  with tf.gfile.GFile(filename) as f:
    for i, line in enumerate(f):
      m = pattern.match(line)
      if m is None:
        logging.warning('incorrect charset file. line #%d: %s', i, line)
        continue
      code = int(m.group(1))
      char = m.group(2)
      if char == '<nul>':
        char = null_character
      charset[code] = char
  return charset


class _NumOfViewsHandler(slim.tfexample_decoder.ItemHandler):
  """Convenience handler to determine number of views stored in an image."""

  def __init__(self, width_key, original_width_key, num_of_views):
    super(_NumOfViewsHandler, self).__init__([width_key, original_width_key])
    self._width_key = width_key
    self._original_width_key = original_width_key
    self._num_of_views = num_of_views

  def tensors_to_item(self, keys_to_tensors):
    return tf.to_int64(
        self._num_of_views * keys_to_tensors[self._original_width_key] /
        keys_to_tensors[self._width_key])


def get_split(class_name, split_name, dataset_dir=None):
  """Returns a dataset tuple for FSNS dataset.

  Args:
    split_name: A train/test split name.
    dataset_dir: The base directory of the dataset sources, by default it uses
      a predefined CNS path (see DEFAULT_DATASET_DIR).

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/test split.
  """
  config = get_config(class_name)

  if split_name not in config['splits']:
    raise ValueError('split name %s was not recognized.' % split_name)

  logging.info('Using %s dataset split_name=%s dataset_dir=%s', config['name'],
               split_name, dataset_dir)

  # Ignores the 'image/height' feature.
  zero = tf.zeros([1], dtype=tf.int64)
  keys_to_features = {
      'image/encoded':
      tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
      tf.FixedLenFeature((), tf.string, default_value='png'),
      'image/width':
      tf.FixedLenFeature([1], tf.int64, default_value=zero),
      'image/orig_width':
      tf.FixedLenFeature([1], tf.int64, default_value=zero),
      'image/class':
      tf.FixedLenFeature([config['max_sequence_length']], tf.int64),
      'image/unpadded_class':
      tf.VarLenFeature(tf.int64),
      'image/text':
      tf.FixedLenFeature([1], tf.string, default_value=''),
  }
  items_to_handlers = {
      'image':
      slim.tfexample_decoder.Image(
          shape=config['image_shape'],
          image_key='image/encoded',
          format_key='image/format'),
      'label':
      slim.tfexample_decoder.Tensor(tensor_key='image/class'),
      'text':
      slim.tfexample_decoder.Tensor(tensor_key='image/text'),
      'num_of_views':
      _NumOfViewsHandler(
          width_key='image/width',
          original_width_key='image/orig_width',
          num_of_views=config['num_of_views'])
  }
  decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                    items_to_handlers)
  charset_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/charset_size.txt')
  charset = read_charset(charset_file)
  file_path = os.path.join(dataset_dir,
                           config['splits'][split_name]['pattern'])

  # auto count number of samples in .tfrecord
  num_samples = len(list(tf.python_io.tf_record_iterator(file_path)))

  return slim.dataset.Dataset(
      data_sources=file_path,
      reader=tf.TFRecordReader,
      decoder=decoder,
      num_samples=num_samples,
      items_to_descriptions=None,
      # additional parameters for convenience.
      charset=charset,
      num_char_classes=len(charset),
      num_of_views=config['num_of_views'],
      max_sequence_length=config['max_sequence_length'],
      null_code=config['null_code'])
