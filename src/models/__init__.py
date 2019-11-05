import os

WORKING_DIR = os.getenv('FTI_IDCARD_HOME') or os.path.abspath('.')
CROPPER_WORKING_DIR = os.path.join(WORKING_DIR, 'src/models/cropper')
DETECTOR_WORKING_DIR = os.path.join(WORKING_DIR,'src/models/detector')