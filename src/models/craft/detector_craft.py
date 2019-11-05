import os
import time
import cv2
import numpy as np
import logging

import torch
import torch.utils.data
from torch.autograd import Variable

# DETECTOR MODEL
from src.models.craft.craft import CRAFT
from src.models.craft import craft_utils
from src.models.craft import imgproc
from collections import OrderedDict

from src.models.craft import craft_utils as craft_utils_char

from src.common import WORKING_DIR


# CRAFT DETECTOR PARAMETERS
PRE_TRAINED_CRAFT_PATH = WORKING_DIR + '/pre-trained_models/craft_mlt_25k.pth'


def copy_state_dict(state_dict):
    # if list(state_dict.keys())[0].startswith("module"):
    #     start_idx = 1
    # else:
    #     start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[1:])
        new_state_dict[name] = v
    return new_state_dict


def load_model():
    """
    Load the pre-trained model, you can use your model just as easily.
    """
    # load model
    logging.info('CRAFT: loading weights from checkpoint (' + PRE_TRAINED_CRAFT_PATH + ')')
    # load CRAFT detector and reader model
    model = CRAFT()
    model.load_state_dict(copy_state_dict(torch.load(PRE_TRAINED_CRAFT_PATH, map_location='cpu')))
    model.eval()

    return model


def test_net(net, image):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, square_size=640,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    with torch.no_grad():
        # pre-processing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]

        # forward pass
        y, _ = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # Post-processing
    boxes, poly = craft_utils.getDetBoxes(score_text, score_link, 0.7, 0.5, 0.4)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    # print(boxes.shape)

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, ret_score_text


def detect(image, model):
    # Using craft: word detector to crop
    bboxes, score_text = test_net(model, image)

    detected_bboxes = []
    for box in bboxes:
        bottom_right = (int(box[np.argmax(np.array(box), axis=0)][0][0]), int(box[np.argmax(np.array(box), axis=0)][1][1]))
        top_left = (int(box[np.argmin(np.array(box), axis=0)][0][0]), int(box[np.argmin(np.array(box), axis=0)][1][1]))
        detected_bboxes.append((top_left, bottom_right))

    #     # Draw for debug
    #     cv2.rectangle(image_detector, top_left, bottom_right, (255,0,0), 2)
    #
    # cv2.imwrite('detected.jpg', image_detector)

    return detected_bboxes


if __name__ == '__main__':
    detect('/home/green/data/samples/samples/03_DL_back.JPG')
