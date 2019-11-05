import datetime
import os
import time
import shutil
import logging

import cv2
import json
import requests
import mimetypes
from pdf2image import convert_from_bytes
import imghdr
import whatimage
import numpy as np
import base64
import hashlib
import traceback
import subprocess

from src.common import WORKING_DIR, UPLOADED_DIR, RESULT_DIR, CROPPED_DIR, FACE_DIR, DEBUG_FLAG, DEBUG_IMAGE
from src.apis import setup_logging

from src.controllers import classifier
from src.controllers import cropper
from src.controllers import detecting_and_reading
from src.controllers import face_detector
from src.controllers import segmentator


# Check if the url image is valid
def is_url_image(url):
    mimetype, _ = mimetypes.guess_type(url)
    return mimetype and mimetype.startswith('image')


class ImageController:
    def __init__(self):
        self.upload_path = ''
        self.result = dict()
        self.result_face = dict()

    def image2text(self, image, data_type, b_face, tf_serving, username, save_debug_images, front_priority,
                   issue_loc_new_back):
        '''
        Extract information from ID card images
        '''
        # Total processing time
        start_time = time.time()

        # Initiate name of image
        filename = ''
        hash_result = ''
        finished_time = 0
        cropped_path = ''
        face_path = ''

        # Image's name with time-stamp
        try:
            if data_type == "url":
                img_name = image.split('/')[-1]
                filename = str(datetime.datetime.now().isoformat()) + '_' + img_name
                if is_url_image(image):
                    try:
                        r = requests.get(image, allow_redirects=True)
                    except:
                        logging.error(traceback.format_exc())
                        self.result['error'] = 'Failed to open the URL!'
                        return self.result, hash_result, filename, finished_time, cropped_path, face_path, self.upload_path, None
                else:
                    self.result['error'] = 'Failed to open the URL!'
                    return self.result, hash_result, filename, finished_time, cropped_path, face_path, self.upload_path, None
            elif data_type == "image":
                # Support receive pdf file
                if image.name.split('.')[-1] == 'pdf':
                    # Convert pdf to image
                    img_pdf = convert_from_bytes(image.body)[0]
                    filename = str(datetime.datetime.now().isoformat()) + '_' + ''.join(image.name.split('.')[:-1]) + '.jpg'
                else:
                    filename = str(datetime.datetime.now().isoformat()) + '_' + image.name
            elif data_type == "masked":
                filename = str(datetime.datetime.now().isoformat()) + '_masked' + '.jpg'
            else:
                filename = str(datetime.datetime.now().isoformat()) + '_base64.jpg'
        except:
            logging.error(traceback.format_exc())
            self.result['error'] = 'Bad data'
            return self.result, hash_result, filename, finished_time, cropped_path, face_path, self.upload_path, None

        # Check if upload path exits
        path_original = os.path.join(UPLOADED_DIR, datetime.date.today().isoformat())
        if not os.path.exists(path_original):
            os.makedirs(path_original, exist_ok=True)
        # Path to the original image
        self.upload_path = os.path.join(path_original, filename)

        # Save original image
        try:
            if data_type == "masked":
                cv2.imwrite(self.upload_path, image)
            else:
                with open(self.upload_path, "wb") as f:
                    if data_type == "url":
                        f.write(r.content)
                    elif data_type == "image":
                        if image.name.split('.')[-1] == 'pdf':
                            img_pdf.save(f)
                        else:
                            f.write(image.body)
                    else:
                        imgdata = base64.b64decode(image)
                        f.write(imgdata)
        except:
            logging.error(traceback.format_exc())

        # Verify that the uploaded image is valid
        try:
            if imghdr.what(self.upload_path) is None:
                with open(self.upload_path, 'rb') as f:
                    data = f.read()
                image_type = whatimage.identify_image(data)
                if image_type is not None and image_type != 'heic':
                    pass
                elif image_type == 'heic' or image_type == 'HEIC':
                    destination = self.upload_path + '.jpg'
                    subprocess.call([WORKING_DIR + '/src/libs/tifig/tifig', '-p', self.upload_path, destination])
                    self.upload_path = destination
                else:   # if not, terminate the process
                    logging.info('{} is not a valid image file'.format(filename))
                    self.result['error'] = 'Invalid image file'
                    return self.result, hash_result, filename, finished_time, cropped_path, face_path, self.upload_path, None
        except IOError:
            logging.error(traceback.format_exc())
            logging.error('Cannot open {}'.format(self.upload_path))
            return

        # If the uploaded image is valid, keep going
        if filename.split('.')[-1].lower() not in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif', 'ppg', 'pgm']:
            filename = filename + '.jpg'
        img_raw = cv2.imread(self.upload_path)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        img_rotated90 = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_rotated_minus90 = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
        img_rotated180 = cv2.rotate(img_rotated90, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        h = hashlib.md5()
        h.update(img_raw)
        hash_result = h.hexdigest()

        ##################################################################################
        # Classifier
        if DEBUG_FLAG:
            tic_classify = time.time()

        img = classifier.read_tensor(img_rgb, input_width=224, input_height=224)

        color_type = classifier.classify_gray_vs_rgb(tf_serving, img) 
        # color_type = 'rgb'
        if DEBUG_FLAG:
            logging.info('Image {} has a {} ID card'.format(filename, color_type))
        
        if color_type == 'rgb':
            card_type, card_type2 = classifier.classify(tf_serving, img)

            tensor_rotated90 = classifier.read_tensor(img_rotated90, input_width=224, input_height=224)
            card_type_90, card_type2_90 = classifier.classify(tf_serving, tensor_rotated90)

            tensor_rotated_minus90 = classifier.read_tensor(img_rotated_minus90, input_width=224, input_height=224)
            card_type_minus90, card_type2_minus90 = classifier.classify(tf_serving, tensor_rotated_minus90)

            tensor_rotated180 = classifier.read_tensor(img_rotated180, input_width=224, input_height=224)
            card_type_180, card_type2_180 = classifier.classify(tf_serving, tensor_rotated180)

        elif color_type == 'gray':
            stacked_img = np.stack((img_gray,)*3, axis=-1)
            img = classifier.read_tensor(stacked_img, input_width=224, input_height=224)
            card_type, card_type2 = classifier.classify_gray(tf_serving, img)

            # tensor_rotated90 = classifier.read_tensor(cv2.cvtColor(img_rotated90, cv2.COLOR_RGB2GRAY), input_width=224, input_height=224)
            # stacked_img = np.stack((tensor_rotated90,)*3, axis=-1)
            # img = classifier.read_tensor(stacked_img, input_width=224, input_height=224)
            # card_type_90, card_type2_90 = classifier.classify_gray(tf_serving, img)

            # tensor_rotated_minus90 = classifier.read_tensor(cv2.cvtColor(img_rotated_minus90, cv2.COLOR_RGB2GRAY), input_width=224, input_height=224)
            # stacked_img = np.stack((tensor_rotated_minus90,)*3, axis=-1)
            # img = classifier.read_tensor(stacked_img, input_width=224, input_height=224)
            # card_type_minus90, card_type2_minus90 = classifier.classify_gray(tf_serving, img)

        # if color_type == 'rgb':
        #     print(card_type, card_type2)
        #     print(card_type_90, card_type2_90)
        #     print(card_type_minus90, card_type2_minus90)
        #     print(card_type_180, card_type2_180)

        if color_type == 'rgb':
            if card_type_90 == card_type_minus90 and card_type != card_type_180 and card_type != card_type_90:
                card_type = card_type_90
            elif card_type_90 == card_type_180 and card_type != card_type_minus90 and card_type != card_type_90:
                card_type = card_type_90
            elif card_type_minus90 == card_type_180 and card_type != card_type_90 and card_type != card_type_minus90:
                card_type = card_type_minus90

            if card_type2_90 == card_type2_minus90 and card_type2 != card_type2_180 and card_type2 != card_type2_90:
                card_type2 = card_type2_90
            elif card_type2_90 == card_type2_180 and card_type2 != card_type2_minus90 and card_type2 != card_type2_90:
                card_type2 = card_type2_90
            elif card_type2_minus90 == card_type2_180 and card_type2 != card_type2_90 and card_type2 != card_type2_minus90:
                card_type2 = card_type2_minus90

        # card_type = 'old'
        # card_type = 'new'
    
        # NOTE: Debug
        if DEBUG_FLAG:
            toc_classify = time.time()
            time_classifying = toc_classify - tic_classify
            logging.info('Image {} is classified as {}'.format(filename, card_type))
            logging.info('{:25s}'.format("Finish CLASSIFYING in: ") + str(time_classifying) + " s")

        ################################################################################################################
        # Two states of cropping 1 and 2(fallback)
        # padding the original image
        # bordersize = 30
        # img_raw=cv2.copyMakeBorder(img_raw, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        next_state = True
        count = 0
        if card_type == 'old':
            count_max = 4
        elif card_type == 'new':
            count_max = 6
        elif card_type == 'old_back':
            count_max = 4
        else:
            count_max = 2 if issue_loc_new_back == '1' else 1

        cropping_state = 1
        logging.info('STATE 1')
        self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_rgb, image_name=filename,
                                                                           card_type=card_type, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
        # print(self.result)
        for k in self.result.keys():
            if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                if float(self.result[k]) >= 2.0:
                    count += 1
        if count >= count_max:
            next_state = False
  
        count = 0
        if (all(value == '' or value in ('N/A', 'NAM', 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 1:
            cropping_state = 2
            logging.info('STATE 2')
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_raw, image_name=filename,
                                                                               card_type=card_type, cropping_state=cropping_state,
                                                                               save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)                                            
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max:
                next_state = False
            
            count = 0

        # padding the original image
        bordersize = 70
        img_padding=cv2.copyMakeBorder(img_raw, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img_rgb_padding = cv2.cvtColor(img_padding, cv2.COLOR_BGR2RGB)

        # if all(value == '' or value == 'N/A' for value in self.result.values()) and cropping_state == 2:
        #     print('STATE 3')
        #     cropping_state = 3
        #     self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_padding, image_name=filename, card_type=card_type, cropping_state=cropping_state, save_debug_images, issue_loc_new_back=issue_loc_new_back)
        
        if (all(value == '' or value in ('N/A', 'NAM', 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 2:
            logging.info('STATE 3')
            cropping_state = 3
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_rgb_padding,
                                                           image_name=filename, card_type=card_type,
                                                           cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)                                                           
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max:
                next_state = False
            count = 0
        
        self.result['type'] = card_type

        ##############
        # Change card_type
        ##############
       
        if card_type2 == 'old':
            count_max2 = 4
        elif card_type2 == 'new':
            count_max2 = 6
        elif card_type2 == 'old_back':
            count_max2 = 4
        else:
            count_max2 = 2 if issue_loc_new_back == '1' else 1

        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 3:
            if DEBUG_FLAG:
                logging.info('Image {} is classified as {}'.format(filename, card_type2))
            logging.info('STATE 1-2')
            cropping_state = 1
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_rgb, image_name=filename, card_type=card_type2, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max2:
                next_state = False
            count = 0
            self.result['type'] = card_type2

        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 1:
            logging.info('STATE 2-2')
            cropping_state = 2
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_raw, image_name=filename, card_type=card_type2, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max2:
                next_state = False
            count = 0
            self.result['type'] = card_type2
     
        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 2:
            logging.info('STATE 3-2')
            cropping_state = 3
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_rgb_padding,
                                                           image_name=filename, card_type=card_type2,
                                                           cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max2:
                next_state = False
            count = 0
            self.result['type'] = card_type2
        
        ##############
        # Rotate the raw image and crop 
        ##############
        # 90 counter clockwise

        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 3:
            logging.info('STATE 1-card_type 1 -rotated 90')
            cropping_state = 1
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_rotated90,
                                                           image_name=filename, card_type=card_type,
                                                           cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max:
                next_state = False
            count = 0
            self.result['type'] = card_type
        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 1:
            logging.info('STATE 2-card_type 1 -rotated 90')
            cropping_state = 2
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, 
                                                           image=cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE),
                                                           image_name=filename, card_type=card_type,
                                                           cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max:
                next_state = False
            count = 0
            self.result['type'] = card_type
        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 2:
            logging.info('STATE 1-card_type 2-rotated 90')
            cropping_state = 1
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_rotated90,
                                                           image_name=filename, card_type=card_type2,
                                                           cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max2:
                next_state = False
            count = 0
            self.result['type'] = card_type2
        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 1:
            logging.info('STATE 2-card_type 2-rotated 90')
            cropping_state = 2
            self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, 
                                                           image=cv2.rotate(img_raw, cv2.ROTATE_90_COUNTERCLOCKWISE),
                                                           image_name=filename, card_type=card_type2,
                                                           cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
            # print(self.result)
            next_state = True
            for k in self.result.keys():
                if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                    if float(self.result[k]) >= 2.0:
                        count += 1
            if count >= count_max2:
                next_state = False
            count = 0
            self.result['type'] = card_type2

        #####
        # After segmenting
        ####
        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 2:
            logging.info('STATE 1 - segmented')
            img_segmented = segmentator.segment(tf_server=tf_serving, image=img_raw, image_name=self.upload_path)
            # print(img_segmented[0])
            # print(len(img_segmented))
 
            if img_segmented is not None and len(img_segmented) > 0:
                try:
                    cropping_state = 1
                    img_segmented_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
                    img_segmented_tensor = classifier.read_tensor(img_segmented_rgb, input_width=224, input_height=224)
                    card_type, card_type2 = classifier.classify(tf_serving, img_segmented_tensor)
                    # print(card_type, card_type2)
                    if card_type == 'old':
                        count_max = 4
                    elif card_type == 'new':
                        count_max = 6
                    elif card_type == 'old_back':
                        count_max = 4
                    else:
                        count_max = 2 if issue_loc_new_back == '1' else 1

                    if card_type2 == 'old':
                        count_max2 = 4
                    elif card_type2 == 'new':
                        count_max2 = 6
                    elif card_type2 == 'old_back':
                        count_max2 = 4
                    else:
                        count_max2 = 2 if issue_loc_new_back == '1' else 1
                    
                    self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_segmented_rgb, image_name=filename, card_type=card_type, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
                    # print(self.result)
                    next_state = True
                    for k in self.result.keys():
                        if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                            if float(self.result[k]) >= 2.0:
                                count += 1
                    if count >= count_max:
                        next_state = False
            
                    count = 0
                    self.result['type'] = card_type
        
                    if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 1:
                        cropping_state = 2
                        logging.info('STATE 2 - segmented')
                        self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_segmented, image_name=filename, card_type=card_type, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
                                                    
                        next_state = True
                        for k in self.result.keys():
                            if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                                if float(self.result[k]) >= 2.0:
                                    count += 1
                        if count >= count_max:
                            next_state = False
                        # print(self.result)
                        count = 0
                        self.result['type'] = card_type
                    
                    if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 2:
                        cropping_state = 1
                        logging.info('STATE 1-2 - segmented')
                        self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_segmented_rgb, image_name=filename, card_type=card_type2, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
                                                    
                        next_state = True
                        for k in self.result.keys():
                            if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                                if float(self.result[k]) >= 2.0:
                                    count += 1
                        if count >= count_max2:
                            next_state = False
                        # print(self.result)
                        count = 0
                        self.result['type'] = card_type2

                    if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 1:
                        cropping_state = 2
                        logging.info('STATE 1-2 - segmented')
                        self.result, cropped_img, cropped_path, points = after_classifying(tf_serving=tf_serving, image=img_segmented, image_name=filename, card_type=card_type2, cropping_state=cropping_state, save_debug_images=save_debug_images, issue_loc_new_back=issue_loc_new_back)
                                                    
                        next_state = True
                        for k in self.result.keys():
                            if k.endswith('prob_old') and self.result[k] not in ('N/A', ''):
                                if float(self.result[k]) >= 2.0:
                                    count += 1
                        if count >= count_max2:
                            next_state = False
                        # print(self.result)
                        count = 0
                        self.result['type'] = card_type2
                except:
                    logging.info(traceback.format_exc())
                    next_state = True
                    cropping_state = 2
        
        if (all(value == '' or value in ('N/A', 'NAM', self.result['type'], 'Failed in cropping') for value in self.result.values()) or next_state) and cropping_state == 2:
            # print('1')
            cropping_state = 1
            self.result['error'] = 'Unable to find ID card in the image'
        elif 'id_prob' in self.result.keys() or 'name' in self.result.keys():
            # print('2')
            if self.result['id_prob_old'] == '0.00' or self.result['id_prob_old'] == '':
                type_dl = classifier.classify_dl(tf_serving, img)
                if type_dl == 'dl new' or type_dl == 'dl old':
                    self.result['error'] = 'Unable to find ID card in the image'
            if len(self.result['name']) >= 30:
                type_dl = classifier.classify_dl(tf_serving, img)
                if type_dl == 'dl new' or type_dl == 'dl old':
                    self.result['error'] = 'Unable to find ID card in the image'
        elif self.result['type'] == 'old_back' or self.result['type'] == 'new_back':
            # print('3')
            if 'issue_date' in self.result.keys() and len(self.result['issue_date']) <= 4:
                type_dl = classifier.classify_dl(tf_serving, img)
                if type_dl == 'dl new' or type_dl == 'dl old':
                    self.result['error'] = 'Unable to find ID card in the image'
        
        # Remove old-style probabilies from the final result
        tmp_result = self.result.copy()
        for k in tmp_result.keys():
            if k.endswith('prob_old'):
                self.result.pop(k)

        if username == 'weshare':
            for key in self.result.keys():
                if not key.endswith('prob') and self.result[key] == '':
                    self.result['error'] = 'Unable to find ID card in the image'
                    break
        ############################
        masked_img = None
        if front_priority != '0':
            if 'error' not in self.result and (self.result['type']=='old_back' or self.result['type']=='new_back'):
                top_left = (int(min([p[0] for p in points])), int(min([p[1] for p in points])))
                bottom_right = (int(max([p[0] for p in points])), int(max([p[1] for p in points])))
                # create mask
                height, width, _ = img_raw.shape
                mask = np.full((height, width, 3),255, dtype='uint8')
                mask = cv2.rectangle(mask, (top_left[0], top_left[1], bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]), color=(0, 0, 0), thickness=-1)
                masked_img = cv2.bitwise_and(src1=img_raw, src2=mask)

        ################################################################################################################
        # Total processing time
        finished_time = time.time() - start_time
        logging.info('{:25s}'.format("Total running time: ") + str(finished_time) + " s")

        if save_debug_images == '1':
            # Check if result path exists
            path_result = os.path.join(RESULT_DIR, datetime.date.today().isoformat())
            if not os.path.exists(path_result):
                os.makedirs(path_result)

            # Result JSON
            result_json = self.result.copy()
            result_json['filename'] = filename
            result_json['username'] = username
            result_json['latency'] = finished_time
            # Result json path
            result_json_path = os.path.join(path_result, filename[:-4] + '.json')
            if not os.path.exists(result_json_path):
                # Create file with JSON enclosures
                with open(result_json_path, 'w') as f:
                    json.dump(result_json, f, ensure_ascii=False)

        if b_face and save_debug_images == '1':
            img = face_detector.crop_face(cropped_img, crop_margin=50, tf_serving=tf_serving)
            if img is None:
               self.result['face'] = 'Unable to find any faces in the image'
            else:
                # Save face image to disk
                path_cropped_face = os.path.join(FACE_DIR, datetime.date.today().isoformat())
                if not os.path.exists(path_cropped_face):
                    os.makedirs(path_cropped_face)
                face_path = os.path.join(path_cropped_face, filename)
                cv2.imwrite(face_path, img)
                self.result['face'] = 'https://static.openfpt.vn/vision'
                for i in face_path.split('/')[-3:]:
                    self.result['face'] = os.path.join(self.result['face'], i)

            self.result['cropped_idcard'] = 'https://static.openfpt.vn/vision'
            for i in cropped_path.split('/')[-3:]:
                self.result['cropped_idcard'] = os.path.join(self.result['cropped_idcard'], i)

        if 'error' in self.result:
            logging.info(username + ' Image ' + filename + ': ' + self.result['error'])
        
        return self.result, hash_result, filename, int(finished_time*1000), cropped_path, face_path, self.upload_path, masked_img


def after_classifying(tf_serving, image, image_name, card_type, cropping_state, save_debug_images, issue_loc_new_back):
    result = dict()

    ##################################################################################
    # Cropper
    if DEBUG_FLAG:
        tic_crop = time.time()

    if cropping_state == 2:
        cropped_img, cropped_list, points = cropper.crop(tf_serving, image, image_name, card_type, save_debug_images)
    else:
        cropped_img, cropped_list, points = cropper.crop_fallback(tf_serving, image, image_name, card_type, save_debug_images)

    if cropped_img is None:
        result['error'] = 'Failed in cropping'
        return result, None, None, None

    if DEBUG_FLAG:
        toc_crop = time.time()
        time_cropping = toc_crop - tic_crop
        logging.info('{:25s}'.format("Finish CROPPING in: ") + str(time_cropping) + " s")

    ##################################################################################
    # Detecting on cropped images

    # Detecting and Reading
    if card_type == 'old' or card_type == 'old_gray':
        result = detecting_and_reading.front('old', cropped_img, image_name, tf_serving)
    elif card_type == 'new' or card_type == 'new_gray':
        result = detecting_and_reading.front('new', cropped_img, image_name, tf_serving)
    elif card_type == 'old_back':
        result = detecting_and_reading.back('old', cropped_img, image_name, tf_serving, issue_loc_new_back)
    elif card_type == 'new_back':
        result = detecting_and_reading.back('new', cropped_img, image_name, tf_serving, issue_loc_new_back)

    if len(cropped_list) > 0:
        return result, cropped_img, cropped_list[0], points 
    else:
        return result, cropped_img, cropped_list, points