import cv2
import glob
import numpy as np
from shutil import copy

con_dau_pos = (320, 240, 270, 240) # (x, y, w, h)

lower_red = np.array([160,100,100])
upper_red = np.array([180,255,255])

# img = cv2.imread('test2.jpg')
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# mask = cv2.inRange(hsv, lower_red, upper_red)
# res = cv2.bitwise_and(img, img, mask=mask)
# cv2.imshow('original', img)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

for f in glob.glob('/home/trung/Projects/ftid/src/models/cropper/test/result/*.jpg'):
    img = cv2.imread(f)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)

    con_dau = mask[con_dau_pos[1]:con_dau_pos[1]+con_dau_pos[3], con_dau_pos[0]:con_dau_pos[0]+con_dau_pos[2]]

    count = 0
    for i in range(con_dau.shape[0]):
        for j in range(con_dau.shape[1]):
            if con_dau[i][j] == 255:
                count += 1

    if count > 20:
        copy(f, '/home/trung/DATA/Backup_zip/MatSau_Cu_cropped/Archive_2/')

