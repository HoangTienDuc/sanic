import numpy as np
import cv2
import tensorflow as tf

def extract_features(img, equalize=True):
    '''
    Calculate histogram gradient of image
    Return numpy array 256-D
    '''
    img = img[50:450, 300:700]
    if equalize:
        img = cv2.equalizeHist(img)
    img = img.astype(np.float32)/255.0
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    sobelx = np.clip(sobelx, -3, 3)
    sobely = np.clip(sobely, -3, 3)
    img_size = img.size
    sobelx = sobelx.flatten()
    sobely = sobely.flatten()
    hogx = np.histogram(sobelx, bins=149, range=(-3, 3))[0]
    hogy = np.histogram(sobely, bins=149, range=(-3, 3))[0]
    hog = np.concatenate([hogx[:64], hogx[85:], hogy[:64], hogy[85:]])
    hog = np.asarray(hog, dtype=np.float32)/(img.size/400.0)
    return hog

def classsify_model(features, labels, mode, params):
    '''Build linear classifier model integrate with estimator'''
    if isinstance(features, dict):
        features = features['feature']
    input_layer = tf.reshape(features, [-1, 256])
    fc1 = tf.keras.layers.Dense(units=4, activation=tf.nn.relu)(input_layer)
    logits = tf.keras.layers.Dense(1)(fc1)
    predicts = tf.nn.sigmoid(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predicts)
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss}, every_n_iter=10)
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
    if mode == tf.estimator.ModeKeys.EVAL:
        thresholds = params['thresholds']
        eval_metric_ops = {
            'precision': tf.metrics.precision_at_thresholds(labels, predicts, thresholds),
            'recall': tf.metrics.recall_at_thresholds(labels, predicts, thresholds),
            'FP': tf.metrics.false_positives_at_thresholds(labels, predicts, thresholds),
            'FN': tf.metrics.false_negatives_at_thresholds(labels, predicts, thresholds)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)
    return classifier
