import os
import cv2
import numpy as np
import argparse


class FeatureCropper(object):

    MAX_FEATURES = 1500
    GOOD_MATCH_PERCENT = 0.15

    WIDTH = 800
    HEIGHT = 500

    RESIZE_SCALE = 1.5   

    def __init__(self, REF_FILE):
        # Preprocess template image
        imRef = cv2.imread(REF_FILE, cv2.IMREAD_COLOR)
        imRef = cv2.GaussianBlur(imRef, (5, 5), 0)
        self.imRef = cv2.cvtColor(imRef, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("Template image", imRef)
        # cv2.waitKey()

        self.orb = cv2.ORB_create(self.MAX_FEATURES)

        self.keypRef, self.descRef = self.orb.detectAndCompute(imRef, None)

        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    def _resize(self, im):
        """Resize images too large/detailed to detect features."""
        height, width, channels = im.shape
        if height >= width and height > self.WIDTH * self.RESIZE_SCALE:
            # print("Resizing image...")
            return cv2.resize(im, (int(self.HEIGHT * self.RESIZE_SCALE), int(self.WIDTH * self.RESIZE_SCALE)), interpolation=cv2.INTER_AREA)
        elif width > height and width > self.WIDTH * self.RESIZE_SCALE:
            # print("Resizing image...")
            return cv2.resize(im, (int(self.WIDTH * self.RESIZE_SCALE), int(self.HEIGHT * self.RESIZE_SCALE)), interpolation=cv2.INTER_AREA)
        else:
            return im

    def align(self, im):
        im = self._resize(im)

        imCopy = im.copy()

        # Preprocess image for better feature detection
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = cv2.equalizeHist(im)

        # cv2.imshow("Query image", im)
        # cv2.waitKey()

        keypoints, descriptors = self.orb.detectAndCompute(im, None)
        matches = self.matcher.match(descriptors, self.descRef)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        # imMatches = cv2.drawMatches(im, keypoints, self.imRef, self.keypRef, matches, None)
        # cv2.imshow("matches", imMatches)

        # Extract location of good matches
        pointsQuery = np.zeros((len(matches), 2), dtype=np.float32)
        pointsRef = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            pointsQuery[i, :] = keypoints[match.queryIdx].pt
            pointsRef[i, :] = self.keypRef[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(pointsQuery, pointsRef, cv2.RANSAC)

        # Use homography
        return cv2.warpPerspective(imCopy, h, (self.WIDTH, self.HEIGHT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="./test/src")
    parser.add_argument("--out_dir", default="./test/result")
    args = parser.parse_args()
    # filenames = [os.path.join(args.in_dir, filename) for filename in os.listdir(args.in_dir)]
    filenames = [filename for filename in os.listdir(args.in_dir)]
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    cropper = FeatureCropper("./template/ref.jpg")
    for filename in filenames:
        # print("Processing {}".format(filename))
        im = cv2.imread(os.path.join(args.in_dir,filename))
        aligned = cropper.align(im)
        cv2.imwrite(os.path.join(args.out_dir, filename), aligned)

if __name__ == '__main__':
    main()