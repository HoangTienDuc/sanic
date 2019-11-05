import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow import app
from tensorflow.python.platform import flags

import data_provider
import common_flags

FLAGS = flags.FLAGS
common_flags.define()

# yapf: disable
flags.DEFINE_integer('num_batches', 100,
                     'Number of batches to run eval for.')

flags.DEFINE_string('eval_log_dir', '/tmp/attention_ocr/eval',
                    'Directory where the evaluation results are saved to.')

flags.DEFINE_integer('eval_interval_secs', 60,
                     'Frequency in seconds to run evaluations.')

flags.DEFINE_integer('number_of_steps', None,
                     'Number of times to run evaluation.')
# yapf: enable


def main(_):
  if not tf.gfile.Exists(FLAGS.eval_log_dir):
    tf.gfile.MakeDirs(FLAGS.eval_log_dir)

  dataset = common_flags.create_dataset(class_name=FLAGS.class_name, split_name=FLAGS.split_name)
  model = common_flags.create_model(dataset.num_char_classes,
                                    dataset.max_sequence_length,
                                    dataset.num_of_views, dataset.null_code)
  data = data_provider.get_data(
      dataset,
      FLAGS.batch_size,
      augment=False,
      central_crop_size=common_flags.get_crop_size())
  endpoints = model.create_base(data.images, labels_one_hot=None)
  model.create_loss(data, endpoints)
  eval_ops = model.create_summaries(
      data, endpoints, dataset.charset, is_training=False)
  slim.get_or_create_global_step()
  #session_config = tf.ConfigProto(device_count={"GPU": 0})
  slim.evaluation.evaluation_loop(
      master=FLAGS.master,
      checkpoint_dir=FLAGS.train_log_dir,
      logdir=FLAGS.eval_log_dir,
      eval_op=list(eval_ops),
      num_evals=FLAGS.num_batches,
      eval_interval_secs=FLAGS.eval_interval_secs,
      max_number_of_evaluations=FLAGS.number_of_steps,)
      #session_config=session_config)


if __name__ == '__main__':
  app.run()
