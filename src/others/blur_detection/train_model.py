import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
from tqdm import tqdm

from .model import classsify_model, extract_features

np.random.seed(217)

def get_dataset(datapath, shuffle=True, batch_size=32, num_epochs=None):
    '''
    Get dataset input fn from datapath

    Parameters
    ----------
    datapath: str
        datapath of directory contain blur directory and not_blur directory image

    Returns
    -------
    dataset input fn
    '''
    blur_img_fns = os.listdir(os.path.join(datapath, 'blur'))
    blur_img_fns = [os.path.join(datapath, 'blur', fn) for fn in blur_img_fns]
    not_blur_img_fns = os.listdir(os.path.join(datapath, 'not_blur'))
    not_blur_img_fns = [os.path.join(datapath, 'not_blur', fn) for fn in not_blur_img_fns]
    dataset_img_fns = blur_img_fns + not_blur_img_fns
    labels = [[1]]*len(blur_img_fns) + [[0]]*len(not_blur_img_fns)
    # dataset_img_fns = dataset_img_fns[:10]
    # labels = labels[:10]
    dataset_indices = [i for i in range(len(dataset_img_fns))]
    img_features = []
    print('Reading dataset:', datapath)
    for img_fn in tqdm(dataset_img_fns):
        img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        img_feature = extract_features(img)
        img_features.append(img_feature)
    img_features = np.asarray(img_features)
    labels = np.asarray(labels)
    return tf.estimator.inputs.numpy_input_fn(img_features, labels, shuffle=True,
        batch_size=batch_size, num_epochs=num_epochs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="batch size for training", default=64)
    parser.add_argument("--steps", default=500)
    parser.add_argument("--train_datapath")
    parser.add_argument("--test_datapath")
    parser.add_argument("--model_dir")
    parser.add_argument("--logpath")
    parser.add_argument("--train", dest='train', action='store_true')
    parser.add_argument("--eval", dest='eval', action='store_true')
    parser.set_defaults(train=False, eval=False)
    args = parser.parse_args()
    batch_size = int(args.batch_size)
    steps = int(args.steps)
    logpath = args.logpath or './logs/default.log'
    train_datapath = args.train_datapath or '/media/trivu/data/DataScience/CV/CMT_full_data/blur_CMT_data/new_front/train'
    test_datapath = args.test_datapath or '/media/trivu/data/DataScience/CV/CMT_full_data/blur_CMT_data/new_front/test'
    model_dir = args.model_dir or './tmp/default'
    train_mode = args.train
    eval_mode = args.eval
    tf.logging.set_verbosity(tf.logging.INFO)
    thresholds = [0.1, 0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    params = {
        'thresholds': thresholds
    }
    classifier = tf.estimator.Estimator(classsify_model, model_dir=model_dir, params=params)
    if train_mode:
        classifier.train(get_dataset(train_datapath, batch_size=batch_size), steps=steps)
    if eval_mode:
        NB_OF_TEST_IMG = len(os.listdir(os.path.join(test_datapath, 'blur')))\
            + len(os.listdir(os.path.join(test_datapath, 'not_blur')))
        evaluation = classifier.evaluate(
            get_dataset(test_datapath, batch_size=batch_size, shuffle=False, num_epochs=1))
        print('--------------------------------------------')
        print('Number of image test:', NB_OF_TEST_IMG)
        print('Evaluation:')
        print('thresholds:', thresholds)
        logfile = open(logpath, 'w')
        logfile.write('thresholds: {}\n'.format(thresholds))
        for key in evaluation:
            print(key, ':', evaluation[key])
            logfile.write('{}: {}\n'.format(key, evaluation[key]))
        logfile.close()

if __name__ == '__main__':
    main()

