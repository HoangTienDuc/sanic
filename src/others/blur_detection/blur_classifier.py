'''
Classify blurry image
'''
import cv2
import numpy as np
from tensorflow.contrib import predictor
import os
from .model import extract_features

BASEDIR = os.path.dirname(os.path.realpath(__file__))

class BlurClassifier:
    '''
    Blur classifier for people ID cards
    '''

    def __init__(self):
        self.new_front_predict_fn = predictor.from_saved_model(os.path.join(BASEDIR, './saved_model/new_front'))
        self.old_front_predict_fn = predictor.from_saved_model(os.path.join(BASEDIR, './saved_model/old_front'))
        self.new_back_predict_fn = predictor.from_saved_model(os.path.join(BASEDIR, './saved_model/new_back'))
        self.old_back_predict_fn = predictor.from_saved_model(os.path.join(BASEDIR, './saved_model/old_back'))

    def _normalize_image(self, img, img_type):
        if img_type == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif img_type == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img_type != 'GRAY':
            raise ValueError('img_type should be RGB, BGR or GRAY')
        if img.shape[0] != 500 or img.shape[1] != 800:
            img = cv2.resize(img, (800, 500))
        return img

    def classify_blur_new_front_id_card(self, img, img_type='RGB', threshold=0.9):
        '''
        Classify blur new front id card

        Parameters
        ----------
        img: numpy array
            Front id card should be cropped and has size (500, 800)
        img_type: str
            One of 'RGB', 'GRAY', 'BGR'

        Returns
        -------
        bool
            return True if image is blurry, otherwise return False
        '''
        img = self._normalize_image(img, img_type)
        features = extract_features(img)
        pred = self.new_front_predict_fn({'input': [features]})['output'][0, 0]
        if pred >= threshold:
            return True
        else:
            return False

    def classify_blur_old_front_id_card(self, img, img_type='RGB', threshold=0.9):
        '''
        Classify blur old front id card

        Parameters
        ----------
        img: numpy array
            Front id card should be cropped and has size (500, 800)
        img_type: str
            One of 'RGB', 'GRAY', 'BGR'

        Returns
        -------
        bool
            return True if image is blurry, otherwise return False
        '''
        img = self._normalize_image(img, img_type)
        features = extract_features(img)
        pred = self.old_front_predict_fn({'input': [features]})['output'][0, 0]
        if pred >= threshold:
            return True
        else:
            return False

    def classify_blur_new_back_id_card(self, img, img_type='RGB', threshold=0.85):
        '''
        Classify blur new back id card

        Parameters
        ----------
        img: numpy array
            Front id card should be cropped and has size (500, 800)
        img_type: str
            One of 'RGB', 'GRAY', 'BGR'

        Returns
        -------
        bool
            return True if image is blurry, otherwise return False
        '''
        img = self._normalize_image(img, img_type)
        features = extract_features(img)
        pred = self.new_back_predict_fn({'input': [features]})['output'][0, 0]
        if pred >= threshold:
            return True
        else:
            return False

    def classify_blur_old_back_id_card(self, img, img_type='RGB', threshold=0.95):
        '''
        Classify blur old back id card

        Parameters
        ----------
        img: numpy array
            Old back id card should be cropped and has size (500, 800)
        img_type: str
            One of 'RGB', 'GRAY', 'BGR'

        Returns
        -------
        bool
            return True if image is blurry, otherwise return False
        '''
        img = self._normalize_image(img, img_type)
        features = extract_features(img)
        pred = self.old_back_predict_fn({'input': [features]})['output'][0, 0]
        if pred >= threshold:
            return True
        else:
            return False
