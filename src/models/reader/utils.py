import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image


def logits_to_log_prob(logits):
  """Computes log probabilities using numerically stable trick.

  This uses two numerical stability tricks:
  1) softmax(x) = softmax(x - c) where c is a constant applied to all
  arguments. If we set c = max(x) then the softmax is more numerically
  stable.
  2) log softmax(x) is not numerically stable, but we can stabilize it
  by using the identity log softmax(x) = x - log sum exp(x)

  Args:
    logits: Tensor of arbitrary shape whose last dimension contains logits.

  Returns:
    A tensor of the same shape as the input, but with corresponding log
    probabilities.
  """

  with tf.variable_scope('log_probabilities'):
    reduction_indices = len(logits.shape.as_list()) - 1
    max_logits = tf.reduce_max(
        logits, reduction_indices=reduction_indices, keep_dims=True)
    safe_logits = tf.subtract(logits, max_logits)
    sum_exp = tf.reduce_sum(
        tf.exp(safe_logits),
        reduction_indices=reduction_indices,
        keep_dims=True)
    log_probs = tf.subtract(safe_logits, tf.log(sum_exp))
  return log_probs


def variables_to_restore(scope=None, strip_scope=False):
  """Returns a list of variables to restore for the specified list of methods.

  It is supposed that variable name starts with the method's scope (a prefix
  returned by _method_scope function).

  Args:
    methods_names: a list of names of configurable methods.
    strip_scope: if True will return variable names without method's scope.
      If methods_names is None will return names unchanged.
    model_scope: a scope for a whole model.

  Returns:
    a dictionary mapping variable names to variables for restore.
  """
  if scope:
    variable_map = {}
    method_variables = slim.get_variables_to_restore(include=[scope])
    for var in method_variables:
      if strip_scope:
        var_name = var.op.name[len(scope) + 1:]
      else:
        var_name = var.op.name
      variable_map[var_name] = var

    return variable_map
  else:
    return {v.op.name: v for v in slim.get_variables_to_restore()}


def resize_image(img, width, height):
  # Resize image to max (width, height), break ratio
  return img.resize((width, height), Image.ANTIALIAS)


def pad_image_keep_ratio(img, width, height):
  # Resize image to max height, keeping aspect ratio
  w, h = img.size
  hpercent = height / float(h)
  wsize = int(w * hpercent)
  if wsize > width:
    # the image is still too long
    # break the ratio anyway
    wsize = width

  img = img.resize((wsize, height), Image.ANTIALIAS)

  # Pad black pixel to max width
  new_img = Image.new('RGB', (width, height))
  new_img.paste(img, ((width - wsize) // 2, 0))

  return new_img
