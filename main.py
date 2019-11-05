import datetime
import os
import logging
import time
import traceback

import boto3
from botocore.client import Config
from multiprocessing import Queue
from threading import Thread

from sanic import Sanic
from sanic.response import json, file

from src.controllers.image_controller import ImageController
from src.apis import setup_logging

# Get config
PREFIX = 'FTI_IDCARD_'
TF_SERVING_HOST = os.getenv(PREFIX + 'SERVING_HOST') or 'localhost:8500'
HOST = os.getenv(PREFIX + 'HOST') or '0.0.0.0'
PORT = os.getenv(PREFIX + 'PORT') or 5000
WORKERS = os.getenv(PREFIX + 'WORKERS') or 1

HOST_NAME = os.getenv('HOST_NAME') or "hostname"
GET_FACE = os.getenv(PREFIX + 'FACE') or '1'
SAVE_DEBUG_IMAGES = os.getenv('SAVE_DEBUG_IMAGES') or '1'
SAVE_NO_CARD_IMAGES = os.getenv('SAVE_NO_CARD_IMAGES') or '1'
FRONT_PRIORITY = os.getenv('FRONT_PRIORITY') or '0'
ISSUE_LOC_NEW_BACK = os.getenv('ISSUE_LOC_NEW_BACK') or '1'


# Setup application
app = Sanic()

# return JSON
ret = dict()

# workers and queue of boto
number_workers = 4
q = Queue()

if GET_FACE == '1' and SAVE_DEBUG_IMAGES == '1':
    s3 = boto3.resource('s3',
                    endpoint_url='https://minio.bcnfpt.com/',
                    aws_access_key_id='vision',
                    aws_secret_access_key='AvNq3wfU0QNrI-PXkqG7',
                    config=Config(signature_version='s3v4'),
                    region_name='ap-southeast-1')

    # each worker does this job
    def pull_from_queue():
        while True:
            item = q.get()
            logging.info('Found {0} in queue'.format(item))
            try:
                print(os.path.join('cropped', datetime.date.today().isoformat(), item[0].split('/')[-1]))
                s3.Bucket('vision').upload_file(item[0], os.path.join('cropped', datetime.date.today().isoformat(), item[0].split('/')[-1]))
            except:
                logging.error(traceback.format_exc())
            try:
                s3.Bucket('vision').upload_file(item[1], os.path.join('faces', datetime.date.today().isoformat(), item[1].split('/')[-1]))
            except:
                logging.info('Unable to find any faces in the image')
                logging.error(traceback.format_exc())
    # init the workers
    for i in range(number_workers):
        t = Thread(target=pull_from_queue)
        t.daemon = True
        t.start()


@app.route('/', methods=['GET', 'OPTIONS'])
async def option(request):
    return json(
        {'success': True},
        headers={
            'Access-Control-Allow-Headers': 'api_key, Content-Type',
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': '*',
            'Hostname': HOST_NAME,
            'X-FPTAI-BILLING': 0
        })


@app.route('/face', methods=['GET'])
async def get_face(request):
    try:
        if request.files is not None and 'face' in request.form:
            path = os.path.join(os.getenv(PREFIX + 'HOME') + "/resources/images/faces", request.form.get('face'))
            if os.path.isfile(path):
                return await file(
                    path,
                    headers={
                        'Access-Control-Allow-Headers': 'api_key, Content-Type',
                        'Content-Type': 'image/jpeg',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': '*',
                        'Hostname': HOST_NAME,
                        'X-FPTAI-BILLING': 0
                    })
            else:
                return json(
                    'No URL in the request',
                    headers={
                        'Access-Control-Allow-Headers': 'api_key, Content-Type',
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': '*',
                        'Hostname': HOST_NAME,
                        'X-FPTAI-BILLING': 0
                    })
        else:
            return json(
                    'Invalid Parameters or Values!',
                    headers={
                        'Access-Control-Allow-Headers': 'api_key, Content-Type',
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': '*',
                        'Hostname': HOST_NAME,
                        'X-FPTAI-BILLING': 0
                    })
    except:
        logging.error(traceback.format_exc())
        return json("Something wrong has happened when getting face",
                    headers={
                        'Access-Control-Allow-Headers': 'api_key, Content-Type',
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': '*',
                        'Hostname': HOST_NAME
                    })


@app.route('/', methods=['POST'])
async def read_api(request):
    logging.info('Started new at: ' + str(datetime.datetime.now()))
    result = dict()
    b_face = False
    hash_result = ''
    filename = ''
    finishing_time = 0
    cropped_path = ''
    face_path = ''
    fptai_billing = 0
    response_status = 200

    result_2 = dict()
    hash_result_2 = ''
    filename_2 = ''
    finishing_time_2 = 0
    cropped_path_2 = ''
    face_path_2 = ''
    masked_img = None

    img_controller = ImageController()

    try:
        if 'x-consumer-username' in request.headers:
            username = request.headers['x-consumer-username']
        else:
            username = ''

        if GET_FACE == '1' and 'face' in request.form and SAVE_DEBUG_IMAGES == '1':
            if request.form.get('face').strip() == '1':
                b_face = True

        if 'image' in request.files:
            image = request.files.get('image')
            data_type = "image"
            result, hash_result, filename, finishing_time, cropped_path, face_path, original_path, masked_img = img_controller.image2text(image, data_type, b_face, TF_SERVING_HOST, username, SAVE_DEBUG_IMAGES, FRONT_PRIORITY, ISSUE_LOC_NEW_BACK)
            if GET_FACE == '1' and 'face' in request.form and SAVE_DEBUG_IMAGES == '1':
                if request.form.get('face').strip() == '1':
                    q.put((cropped_path, face_path))
        elif 'image_url' in request.form:
            url = request.form.get('image_url')
            if url != '':
                data_type = 'url'
                result, hash_result, filename, finishing_time, cropped_path, face_path, original_path, masked_img = img_controller.image2text(url, data_type, b_face, TF_SERVING_HOST, username, SAVE_DEBUG_IMAGES, FRONT_PRIORITY, ISSUE_LOC_NEW_BACK)
                if GET_FACE == '1' and 'face' in request.form and SAVE_DEBUG_IMAGES == '1':
                    if request.form.get('face').strip() == '1':
                        q.put((cropped_path, face_path))
            else:
                result['error'] = 'No URL in the request'
        elif 'image_base64' in request.form:
            img_base64 = request.form.get('image_base64')
            if img_base64 != '':
                if len(img_base64.strip()) % 4 == 0:
                    data_type = 'base64'
                    result, hash_result, filename, finishing_time, cropped_path, face_path, original_path, masked_img = img_controller.image2text(img_base64, data_type, b_face, TF_SERVING_HOST, username, SAVE_DEBUG_IMAGES, FRONT_PRIORITY, ISSUE_LOC_NEW_BACK)
                    if GET_FACE == '1' and 'face' in request.form and SAVE_DEBUG_IMAGES == '1':
                        if request.form.get('face').strip() == '1':
                            q.put((cropped_path, face_path))
                else:
                    result['error'] = 'String base64 is not valid'
            else:
                result['error'] = 'No string base64 in the request'
        else:
            result['error'] = 'Invalid Parameters or Values!'

        if masked_img is not None:
            data_type = "masked"
            result_2, hash_result_2, filename_2, finishing_time_2, cropped_path_2, face_path_2, original_path_2, masked_img_2 = img_controller.image2text(masked_img, data_type, b_face, TF_SERVING_HOST, username, SAVE_DEBUG_IMAGES, FRONT_PRIORITY, ISSUE_LOC_NEW_BACK)
            if GET_FACE == '1' and 'face' in request.form and SAVE_DEBUG_IMAGES == '1':
                if request.form.get('face').strip() == '1':
                    q.put((cropped_path_2, face_path_2))

        # Format the return
        if 'error' in result:
            fptai_billing = 1
            if result['error'] == 'Invalid Parameters or Values!':
                ret['errorCode'] = 1
                ret['errorMessage'] = result['error']
                ret['data'] = []
                response_status = 400
            elif result['error'] == 'Failed in cropping':
                ret['errorCode'] = 2
                ret['errorMessage'] = result['error']
                ret['data'] = []
            elif result['error'] == 'Unable to find ID card in the image':
                if SAVE_NO_CARD_IMAGES != '1':
                    os.remove(original_path)
                ret['errorCode'] = 3
                ret['errorMessage'] = result['error']
                ret['data'] = []
            # code 4: Backside has not been supported
            elif result['error'] == 'No URL in the request':
                ret['errorCode'] = 5
                ret['errorMessage'] = result['error']
                ret['data'] = []
                response_status = 400
            elif result['error'] == 'Failed to open the URL!':
                ret['errorCode'] = 6
                ret['errorMessage'] = result['error']
                ret['data'] = []
            elif result['error'] == 'Invalid image file':
                ret['errorCode'] = 7
                ret['errorMessage'] = result['error']
                ret['data'] = []
            elif result['error'] == 'Bad data':
                ret['errorCode'] = 8
                ret['errorMessage'] = result['error']
                ret['data'] = []
            elif result['error'] == 'No string base64 in the request':
                ret['errorCode'] = 9
                ret['errorMessage'] = result['error']
                ret['data'] = []
                response_status = 400
            elif result['error'] == 'String base64 is not valid':
                ret['errorCode'] = 10
                ret['errorMessage'] = result['error']
                ret['data'] = []
        else:
            if result['type'] == 'old' or result['type'] == 'new':
                fptai_billing = 1

            ret['errorCode'] = 0
            ret['errorMessage'] = ''
            if 'error' in result_2 or result_2 == {}:
                ret['data'] = [result]
            else:
                ret['data'] = [result, result_2]

        return json(ret, ensure_ascii=False, escape_forward_slashes=False,
                    headers={
                        'Access-Control-Allow-Headers': 'api_key, Content-Type',
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': '*',
                        'Hostname': HOST_NAME,
                        'MD5': hash_result,
                        'File-Name': filename,
                        'Process-Time': finishing_time + finishing_time_2,
                        'X-FPTAI-BILLING': fptai_billing
                    },
                    status=response_status)
    except:
        logging.error(traceback.format_exc())
        return json("Something wrong has happened",
                    headers={
                        'Access-Control-Allow-Headers': 'api_key, Content-Type',
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': '*',
                        'Hostname': HOST_NAME,
                        'MD5': hash_result,
                        'File-Name': filename,
                        'Process-Time': finishing_time + finishing_time_2,
                        'X-FPTAI-BILLING': fptai_billing
                    },
                    status=500)

if __name__ == '__main__':
    app.run(port=int(PORT), host=HOST, workers=int(WORKERS), debug=False)
