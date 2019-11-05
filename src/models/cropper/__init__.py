import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import os
import time
import cv2
import numpy as np
import argparse

from src.models.cropper.encoder import DataEncoder
from src.models.cropper.retinanet import RetinaNet
from src.models.cropper.feature_cropper import FeatureCropper

WORKING_DIR = os.environ['FTI_IDCARD_HOME'] + '/src/models/cropper'
CHECKPOINT_FILE_PATH = WORKING_DIR + '/checkpoint/corners.pth'
REF_FILE = WORKING_DIR + "/template/ref.jpg"
DEBUG = True

def order(boxes):
    tlbr = np.sum(boxes, axis=1)
    trbl = np.sum(boxes * np.array([-1, 1, -1, 1]), axis=1)
    tl = np.argmin(tlbr)
    br = np.argmax(tlbr)
    tr = np.argmin(trbl)
    bl = np.argmax(trbl)
    return tl, tr, bl, br


def orient_90(boxes):
    tl, tr, bl, br = order(boxes)
    wd = np.max(boxes[:, 2]) - np.min(boxes[:, 0])
    ht = np.max(boxes[:, 3]) - np.min(boxes[:, 1])
    if wd > ht:
        return 0
    else:
        return 90


def crop(image, box):
    return image[box[1]:box[3], box[0]:box[2]]


def orient_180(image):
    test_box = np.array([85, 10, 210, 180])
    inv_box = np.array([image.shape[1] - test_box[2],
                        image.shape[0] - test_box[3],
                        image.shape[1] - test_box[0],
                        image.shape[0] - test_box[1]])
    red = image[:, :, 2].astype(int) - image[:, :, 1].astype(int)
    if np.sum(crop(red, test_box)) > np.sum(crop(red, inv_box)):
        return 0
    else:
        return 180


def target_corners(target_size):
    return np.array([[0, 0],
                     [target_size[0] - 1, 0],
                     [0, target_size[1] - 1],
                     [target_size[0] - 1, target_size[1] - 1]],
                    dtype=np.float32)


class ImageCropper(object):
    def __init__(self, save_file= CHECKPOINT_FILE_PATH , batch_size=100, use_cuda=False):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.net = RetinaNet()
        self.net.load_state_dict(torch.load(save_file))
        if self.use_cuda:
            self.net.cuda()
        self.net.eval()
        self.batch_size = batch_size
        frac = 5 / 6
        self.margins = np.array([[frac, frac, 1 - frac, 1 - frac],
                                 [1 - frac, frac, frac, 1 - frac],
                                 [frac, 1 - frac, 1 - frac, frac],
                                 [1 - frac, 1 - frac, frac, frac]])
        self.target_size = [800, 500]
        self.encoder = DataEncoder()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.feature_cropper = FeatureCropper(REF_FILE)

    def feature_crop(self, filename, image, out_dir):
        """Use feature matching to crop IDs from image"""
        aligned = self.feature_cropper.align(image)
        if DEBUG:
            print("Finish processing with CV cropper and saving to disk")
        cv2.imwrite(os.path.join(out_dir, os.path.basename(filename)), aligned)

    def detect(self, images):
        x = []
        result = []
        w = h = 600
        for image in images:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Preferable method of interpolation for shrinking is cv2.INTER_AREA,
            # but this is consistent with previous implementation.
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
            x.append(self.transform(img))
        x = torch.stack(x)
        if self.use_cuda:
            x = x.cuda()
        # UserWarning: volatile was removed and now has no effect. Use `with torch.no_grad():` instead.
        # Also Variable wrapper is now unnecessary.
        x = Variable(x, volatile=True)
        loc, cls = self.net(x)
        
        for loc_preds, cls_preds in zip(loc.cpu(), cls.cpu()):
            boxes, labels = self.encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
            result.append((boxes, labels))

        return result

    def process_batch(self, filenames, out_dir, rotation=0):
        rotation_dict = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        r90 = []
        images = [cv2.imread(filename) for filename in filenames]
        idx = [i for i in range(len(images)) if images[i] is not None]
        images = [images[i] for i in idx]
        filenames = [filenames[i] for i in idx]
        if rotation != 0:
            images = [cv2.rotate(image, rotation_dict[rotation]) for image in images]
            # for image in images:
            #     cv2.imshow("rotating image", image)
            #     cv2.waitKey()
        for filename, image, (boxes, labels) in zip(filenames, images, self.detect(images)):
            # print("Processing {} on rotation {}".format(filename, rotation))
            if boxes.size() != torch.Size([4, 4]):
                # Early exits when rotated objects are not detected
                # possibly due to network trained to detect correctly oriented IDs only
                self.feature_crop(filename, image, out_dir)
                continue
            boxes = boxes.numpy() * [image.shape[1] / 600,
                                     image.shape[0] / 600,
                                     image.shape[1] / 600,
                                     image.shape[0] / 600]
            angle = orient_90(boxes)
            if angle == 90:
                r90.append(filename)
                continue
            ordered = list(order(boxes))
            if len(np.unique(ordered)) != 4:               
                self.feature_crop(filename, image, out_dir)
                continue
            boxes = boxes[ordered]
            corners = [None] * 4
            for label, box in zip(np.arange(4), boxes):
                dot = box * self.margins[label]
                dot = [dot[0] + dot[2], dot[1] + dot[3]]
                corners[label] = dot

            if None in corners:                
                self.feature_crop(filename, image, out_dir)
                continue
            else:
                corners = np.array(corners, dtype=np.float32)

            warper = cv2.getPerspectiveTransform(corners, target_corners(self.target_size))
            image = cv2.warpPerspective(image, warper, tuple(self.target_size))
            if orient_180(image) == 180:
                image = cv2.rotate(image, rotation_dict[180])
            # cv2.imshow("image", image)
            # cv2.waitKey()

            # Saving image to disk wastes time, considerably more so than
            # passing it along to the next part of the pipeline
            # TODO: modify the pipeline to pass the image data along on RAM
            if DEBUG:
                print("Finish processing with RetinaNet and saving to disk")
                
            cv2.imwrite(os.path.join(out_dir, os.path.basename(filename)), image)
        return r90

    def process_images(self, filenames, out_dir):
        rotated = []
        for i in range(0, len(filenames), self.batch_size):
            rotated.extend(self.process_batch(filenames[i:i + self.batch_size], out_dir, rotation=0))
        for i in range(0, len(rotated), self.batch_size):
            self.process_batch(rotated[i:i + self.batch_size], out_dir, rotation=90)

    def process_image(self, filename, out_dir):
        rotated = []
        rotated.extend(self.process_batch([filename], out_dir, rotation=0))
        if len(rotated) > 0:
            self.process_batch(rotated, out_dir, rotation=90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default=WORKING_DIR+"/test/src")
    parser.add_argument("--out_dir", default=WORKING_DIR+"/test/result")
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()
    print(args)

    DEBUG = args.debug
    
    filenames = [os.path.join(args.in_dir, filename) for filename in os.listdir(args.in_dir)]
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    cropper = ImageCropper(batch_size=1)
    cropper.process_images(filenames, args.out_dir)
