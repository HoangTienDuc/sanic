import sys
import logging

from tensorflow.python.platform import flags

import dataset
import model


FLAGS = flags.FLAGS

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stderr,
    format='%(levelname)s '
    '%(asctime)s.%(msecs)06d: '
    '%(filename)s: '
    '%(lineno)d '
    '%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def define():
  """Define common flags."""
  # Must
  flags.DEFINE_string('class_name', 'all',
                      'Class to train (name, id, dob, etc.)')

  flags.DEFINE_string('dataset_dir', None,
                      'Dataset root folder.')

  # Optional
  flags.DEFINE_string('split_name', 'train',
                      'Dataset split name to run evaluation for: test,train.')

  flags.DEFINE_integer('batch_size', 32,
                       'Batch size.')

  flags.DEFINE_integer('crop_width', None,
                       'Width of the central crop for images.')

  flags.DEFINE_integer('crop_height', None,
                       'Height of the central crop for images.')

  flags.DEFINE_string('train_log_dir', '/tmp/attention_ocr/train',
                      'Directory where to write event logs.')

  flags.DEFINE_string('checkpoint', '',
                      'Path for checkpoint to restore weights from.')

  flags.DEFINE_string('master',
                      '',
                      'BNS name of the TensorFlow master to use.')

  # Model hyper parameters
  flags.DEFINE_float('learning_rate', 0.004,
                     'learning rate')

  flags.DEFINE_string('optimizer', 'momentum',
                      'the optimizer to use')

  flags.DEFINE_float('momentum', 0.9,
                      'momentum value for the momentum optimizer if used')

  flags.DEFINE_bool('use_augment_input', False,
                    'If True will use image augmentation')

  # Method hyper parameters
  # conv_tower_fn
  flags.DEFINE_string('final_endpoint', 'Mixed_6a',
                      'Endpoint to cut inception tower')

  flags.DEFINE_bool('freeze_cnn', False,
                    'If True then the weights of CNN tower is not updated, used for finetune')

  # sequence_logit_fn
  flags.DEFINE_bool('use_attention', True,
                    'If True will use the attention mechanism')

  flags.DEFINE_bool('use_autoregression', True,
                    'If True will use autoregression (a feedback link)')

  flags.DEFINE_integer('num_lstm_units', 256,
                       'number of LSTM units for sequence LSTM')

  flags.DEFINE_float('weight_decay', 0.00004,
                     'weight decay for char prediction FC layers')

  flags.DEFINE_float('lstm_state_clip_value', 10.0,
                     'cell state is clipped by this value prior to the cell'
                     ' output activation')

  flags.DEFINE_float('label_smoothing', 0.1,
                     'weight for label smoothing')

  flags.DEFINE_bool('ignore_nulls', True,
                    'ignore null characters for computing the loss')

  flags.DEFINE_bool('average_across_timesteps', False,
                    'divide the returned cost by the total label weight')


def get_crop_size():
  if FLAGS.crop_width and FLAGS.crop_height:
    return (FLAGS.crop_width, FLAGS.crop_height)
  else:
    return None


def create_dataset(class_name, split_name):
  if FLAGS.dataset_dir == None:
    print('ERROR: flag --dataset_dir (contains charaset_size.txt, *.tfrecord) must be specified')
    sys.exit(1)

  if FLAGS.class_name == None:
    print('ERROR: flag --class_name (name, id, dob, etc.) must be specified')
    sys.exit(1)

  return dataset.get_split(class_name, split_name,
                           dataset_dir=FLAGS.dataset_dir)


def create_mparams():
  return {
      'conv_tower_fn':
      model.ConvTowerParams(final_endpoint=FLAGS.final_endpoint, is_training=not FLAGS.freeze_cnn),
      'sequence_logit_fn':
      model.SequenceLogitsParams(
          use_attention=FLAGS.use_attention,
          use_autoregression=FLAGS.use_autoregression,
          num_lstm_units=FLAGS.num_lstm_units,
          weight_decay=FLAGS.weight_decay,
          lstm_state_clip_value=FLAGS.lstm_state_clip_value),
      'sequence_loss_fn':
      model.SequenceLossParams(
          label_smoothing=FLAGS.label_smoothing,
          ignore_nulls=FLAGS.ignore_nulls,
          average_across_timesteps=FLAGS.average_across_timesteps)
  }


def create_model(*args, **kwargs):
  ocr_model = model.Model(mparams=create_mparams(), *args, **kwargs)
  return ocr_model
