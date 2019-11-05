import cv2
import glob
import os

folder = '/home/trung/Projects/ftid/src/models/cropper/test/src'
for f in os.listdir('/home/trung/Projects/ftid/src/models/cropper/test/result'):
    file_path = os.path.join(folder, f)
    try:
        if os.path.isfile(file_path):
            print(file_path)
            os.remove(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
