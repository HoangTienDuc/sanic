#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import hashlib
import os

import io

WORKING_DIR = os.getenv('FTI_IDCARD_HOME') or os.path.abspath('.')
MODE_WRITE = 'w'
MODE_READ = 'r'
MODE_RUNTIME = 'test'
CONFIG_FILE_PATH = WORKING_DIR + '/resources/configs/main.conf'
LOG_CONFIG_FILE_PATH = WORKING_DIR + '/resources/configs/logging.conf'
LOG_FILE_PATH = WORKING_DIR + '/logs/apis.log'

UPLOADED_DIR = WORKING_DIR + '/resources/images/original'
CROPPED_DIR = WORKING_DIR + '/resources/images/cropped'
RESULT_DIR = WORKING_DIR + '/resources/images/result'
DEBUG_DIR = WORKING_DIR + '/resources/images/saved'
FACE_DIR = WORKING_DIR + '/resources/images/faces'

DEBUG_FLAG = True
DEBUG_IMAGE = False

BATCH_SIZE = 4


# Classes of reader's names
class ReaderOld:
    ID = 'id-reader'
    NAME = 'name-reader'
    DOB = 'birthday-reader'
    ADD = 'address-reader'
    FEATURE = 'feature-old-back-reader'


class ReaderNew:
    ID = 'id-new-reader'
    NAME = 'name-new-reader'
    DOB = 'birthday-new-reader'
    SEX = 'sex-new-reader'  # Same as address reader
    RACE = 'race-new-reader'  # Same as address reader
    ADD = 'address-new-reader'
    HOME = 'home-new-reader'  # Same as address reader
    FEATURE = 'feature-new-back-reader'
    ISSUE_DATE = 'date-new-back-reader'

