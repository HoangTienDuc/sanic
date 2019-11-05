import os
import json
import logging.config

WORKING_DIR = os.getenv('FTI_IDCARD_HOME') or os.path.abspath('.')
path = os.path.join(WORKING_DIR, 'resources/configs/logging_conf.json')

if os.path.exists(path):
    with open(path, 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
else:
    logging.basicConfig(level=logging.INFO)

