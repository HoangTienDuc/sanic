import os
import datetime
import time
import cv2
import logging
import traceback

from src.controllers import detector
from src.controllers import reader
from src.controllers import post_processing
from src.controllers import issue_loc

from src.common import DEBUG_DIR, DEBUG_FLAG, DEBUG_IMAGE, ReaderOld, ReaderNew
from src.apis import setup_logging

from src.others.correct_addr.address_correction import AddressCorrection
from src.others.address_standardization.address_correction import AddressCorrection as AddressStandardization
from src.others.correct_id.id_correction import IDCorrection
from src.others.correct_rel_eth.religion_correction import ReligionCorrection
from src.others.correct_rel_eth.ethnic_correction import EthnicCorrection

address_correction = AddressCorrection()
id_correction = IDCorrection()
religion_correction = ReligionCorrection()
ethnic_correction = EthnicCorrection()

address_standardization = AddressStandardization()

# draw bounding box for debug
ENABLE_DRAW = False

# Debug directory of the day
path_debug = ''
if DEBUG_IMAGE:
    path_debug = os.path.join(DEBUG_DIR, datetime.date.today().isoformat())
    os.makedirs(path_debug, exist_ok=True)


def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,
                               class_type,
                               use_normalized_coordinates=True):
    if use_normalized_coordinates:
        (left, right, top, bottom) = (int(xmin * image.shape[1]), int(xmax * image.shape[1]),
                                      int(ymin * image.shape[0]), int(ymax * image.shape[0]))
    else:
        (left, right, top, bottom) = (int(xmin), int(xmax), int(ymin), int(ymax))
    if class_type == 1:  # id - green
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    elif class_type == 2:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 2)
    elif class_type == 3:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_type == 4:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)
    elif class_type == 5:
        cv2.rectangle(image, (left, top), (right, bottom), (122, 255, 122), 2)
    elif class_type == 6:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 0), 2)
    elif class_type == 7:
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 255), 2)
    elif class_type == 8:
        cv2.rectangle(image, (left, top), (right, bottom), (122, 122, 255), 2)

    return image


def front(id_type, img, img_name, tf_serving):
    """

    :param img:
    :param img_name:
    :param tf_serving:
    :return:
    """
    if DEBUG_FLAG:
        tic_detector = time.time()

    # Detecting
    if id_type == 'old':
        detected_all_fields = detector.detect(tf_serving, 'all-fields-detector', img)
        batch_size = len(detected_all_fields['scores'])
    else:  # new
        detected_all_fields = detector.detect(tf_serving, 'all-fields-new-detector', img)
        detected_id_new = detector.detect(tf_serving, 'id-new-detector', img)
        detected_date_new = detector.detect(tf_serving, 'date-new-detector', img)
        batch_size = len(detected_id_new['scores'])

    if DEBUG_FLAG:
        toc_detector = time.time()
        logging.info('{:25s}'.format("Finish DETECTING in: ") + str(toc_detector - tic_detector) + " s")

    if DEBUG_FLAG:
        tic_reader = time.time()

    name_coords = list()
    address_coords = list()
    home_coords = list()
    race_coords = list()
    id_coords = list()
    dob_coords = list()
    doe_coords = list()
    sex_coord = None

    # Collect detected boxes
    for i in range(batch_size):
        for j in range(100):
            if detected_all_fields['classes'][i][j] == 2 and detected_all_fields['scores'][i][j] >= 0.25:  # name
                name_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields["boxes"][i][j]))
                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, name_coords[-1][0][1],
                                                     name_coords[-1][0][0],
                                                     name_coords[-1][1][1],
                                                     name_coords[-1][1][0],
                                                     class_type=2,
                                                     use_normalized_coordinates=False)

            if (id_type == 'old' and
                detected_all_fields['classes'][i][j] == 4 and
                detected_all_fields['scores'][i][j] >= 0.35) or (id_type == 'new' and
                                                                 detected_all_fields['classes'][i][j] == 7 and
                                                                 detected_all_fields['scores'][i][j] >= 0.3):  # address
                address_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                # print(address_coords)
                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, address_coords[-1][0][1],
                                                     address_coords[-1][0][0],
                                                     address_coords[-1][1][1],
                                                     address_coords[-1][1][0],
                                                     class_type=4,
                                                     use_normalized_coordinates=False)

            if id_type == 'new' and detected_all_fields['classes'][i][j] == 6 and detected_all_fields['scores'][i][
                j] >= 0.4:  # home
                home_coords.append(reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, home_coords[-1][0][1],
                                                     home_coords[-1][0][0],
                                                     home_coords[-1][1][1],
                                                     home_coords[-1][1][0],
                                                     class_type=8,
                                                     use_normalized_coordinates=False)

            if id_type == 'new' and detected_all_fields['classes'][i][j] == 5 and detected_all_fields['scores'][i][
                j] >= 0.4:  # race
                race_coords.append(reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, race_coords[-1][0][1],
                                                     race_coords[-1][0][0],
                                                     race_coords[-1][1][1],
                                                     race_coords[-1][1][0],
                                                     class_type=7,
                                                     use_normalized_coordinates=False)

            if id_type == 'new' and detected_all_fields['classes'][i][j] == 4 and detected_all_fields['scores'][i][
                j] >= 0.4:  # sex
                sex_coord = reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j])
                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, sex_coord[0][1],
                                                     sex_coord[0][0],
                                                     sex_coord[1][1],
                                                     sex_coord[1][0],
                                                     class_type=6,
                                                     use_normalized_coordinates=False)

            if (id_type == 'old' and
                detected_all_fields['classes'][i][j] == 1 and
                detected_all_fields['scores'][i][j] >= 0.2) or (id_type == 'new' and
                                                                detected_id_new['classes'][i][j] == 1 and
                                                                detected_id_new['scores'][i][j] >= 0.2):  # id
                if id_type == 'old':
                    id_coords.append(reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                else:
                    id_coords.append(reader.convert_bbox_coordinates(img, detected_id_new['boxes'][i][j]))
                    # print(id_coords)

                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, id_coords[-1][0][1],
                                                     id_coords[-1][0][0],
                                                     id_coords[-1][1][1],
                                                     id_coords[-1][1][0],
                                                     class_type=1,
                                                     use_normalized_coordinates=False)

            if (id_type == 'old' and
                detected_all_fields['classes'][i][j] == 3 and
                detected_all_fields['scores'][i][j] >= 0.2) or (id_type == 'new' and
                                                                detected_date_new['classes'][i][j] == 1 and
                                                                detected_date_new['scores'][i][j] >= 0.2):  # dob
                if id_type == 'old':
                    dob_coords.append(reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                else:
                    dob_coords.append(reader.convert_bbox_coordinates(img, detected_date_new['boxes'][i][j]))
                # print(dob_coords)

                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, dob_coords[-1][0][1],
                                                     dob_coords[-1][0][0],
                                                     dob_coords[-1][1][1],
                                                     dob_coords[-1][1][0],
                                                     class_type=3,
                                                     use_normalized_coordinates=False)

            if id_type == 'new' and detected_date_new['classes'][i][j] == 2 and detected_date_new['scores'][i][
                j] >= 0.15:  # doe
                doe_coords.append(reader.convert_bbox_coordinates(img, detected_date_new['boxes'][i][j]))
                # print(doe_coords)
                if ENABLE_DRAW:
                    img = draw_bounding_box_on_image(img, doe_coords[-1][0][1],
                                                     doe_coords[-1][0][0],
                                                     doe_coords[-1][1][1],
                                                     doe_coords[-1][1][0],
                                                     class_type=5,
                                                     use_normalized_coordinates=False)

    # Reading
    res = {}

    id_result = {
        'result': ''
    }
    for i in range(len(id_coords)):
        result = reader.read(tf_serving,
                             reader_name=ReaderOld.ID if id_type == 'old' else ReaderNew.ID,
                             img=img,
                             filename=img_name,
                             box_coord=id_coords[i],
                             save_dir=path_debug)
        # print(result)
        tmp_id = ''
        for letter in result['result']:
            if not (letter.isalpha()):
                tmp_id += letter
        result['result'] = tmp_id
        if result['result'] != '':
            id_result = result

    # Re-scaling the prob
    if id_result['result']:
        prob_id = id_result['prob'] + 0.07
        while prob_id > 1.0:
            prob_id = prob_id - 0.05
        id_result['prob'] = prob_id

    res.update({
        'id': id_result['result'],
        'id_prob': id_result['prob'] if id_result['result'] else '',
        'id_prob_old': id_result['prob_old_style'] if id_result['result'] else ''
    })

    name_result = post_processing.post_process_name(img,
                                                    img_name,
                                                    name_coords,
                                                    ReaderOld.NAME if id_type == 'old' else ReaderNew.NAME,
                                                    tf_serving,
                                                    path_debug)
    res.update({
        'name': name_result['result'],
        'name_prob': name_result['prob'] if name_result['result'] else '',
        'name_prob_old': name_result['prob_old_style'] if name_result['result'] else ''
    })

    dob_result = {
        'result': ''
    }
    for i in range(len(dob_coords)):
        result = reader.read(tf_serving,
                             reader_name=ReaderOld.DOB if id_type == 'old' else ReaderNew.DOB,
                             img=img,
                             filename=img_name,
                             box_coord=dob_coords[i],
                             save_dir=path_debug)
        tmp_date = ''
        # print(result)
        for letter in result['result']:
            if not (letter.isalpha()):
                tmp_date += letter
        result['result'] = tmp_date
        if result['result'] != '':
            dob_result = result
    # print(dob_result)

    # Re-scaling the prob
    if dob_result['result']:
        prob_dob = dob_result['prob'] + 0.05
        while prob_dob > 1.0:
            prob_dob = prob_dob - 0.05
        dob_result['prob'] = prob_dob

    res.update({
        'dob': dob_result['result'].replace('-', '/'),
        'dob_prob': dob_result['prob'] if dob_result['result'] else '',
        'dob_prob_old': dob_result['prob_old_style'] if dob_result['result'] else ''
    })

    if id_type == 'old':
        res.update({
            'sex': 'N/A',
            'sex_prob': 'N/A',
            'sex_prob_old': 'N/A',
            'nationality': 'N/A',
            'nationality_prob': 'N/A',
            'nationality_prob_old': 'N/A'
        })
        address_result, home_result = post_processing.post_process_address_old(img,
                                                                               img_name,
                                                                               address_coords,
                                                                               reader_name=ReaderOld.ADD,
                                                                               tf_serving=tf_serving,
                                                                               save_dir=path_debug)
        res.update({
            'home': home_result['result'],
            'home_prob': home_result['prob'] if home_result['result'] else '',
            'home_prob_old': home_result['prob_old_style'] if home_result['result'] else '',
            'address': address_result['result'],
            'address_prob': address_result['prob'] if address_result['result'] else '',
            'address_prob_old': address_result['prob_old_style'] if address_result['result'] else '',
            'type_new': 'cmnd_09_front'
        })

        # Extract address entities
        address_entities = address_standardization.extract_address_entity(address_result['result'].replace(',', ''))
        res.update({
            'address_entities': address_entities
        })

        res.update({
            'doe': 'N/A',
            'doe_prob': 'N/A',
            'doe_prob_old': 'N/A'
        })
        # print(res['id'])
        # print(res['id_prob'])
        # print(res['address_prob'])
        # print(len(res['id'].strip()))
        # Correct ID
        try:
            if len(res['id'].strip()) < 9 or (id_result['prob_old_style'] < 0.4 and res['address_prob'] > 0.1):
                # print('INSIDE')
                res.update({
                    'id': id_correction.correction_old_id(res['address'], res['id'])
                })
                # print(res['id'])
        except:
            logging.info(traceback.format_exc())
            pass

    else:  # new
        sex_result = reader.read_batch(tf_serving,
                                       reader_name=ReaderNew.ADD,
                                       img=img,
                                       filename='sex_{}'.format(img_name),
                                       box_coords=[sex_coord],
                                       save_dir=path_debug)

        if len(sex_result) == 0:
            res.update({
                'sex': '',
                'sex_prob': '',
                'sex_prob_old': ''
            })
        else:
            # just get first result from batch ones
            sex, sex_prob, prob_sex_old_style = sex_result[0]

            # Re-scaling the prob
            prob_sex = sex_prob + 0.05
            while prob_sex > 1.0:
                prob_sex = prob_sex - 0.05
            sex_prob = prob_sex

            res.update({
                'sex': sex,
                'sex_prob': sex_prob,
                'sex_prob_old': prob_sex_old_style
            })

        race_result = post_processing.post_process_single_line(img,
                                                               img_name,
                                                               race_coords,
                                                               reader_name=ReaderNew.ADD,
                                                               tf_serving=tf_serving,
                                                               save_dir=path_debug)

        if race_result['result'].find('VIỆT') != -1 or race_result['result'].find('NAM') != -1 or race_result[
            'result'].find('VI') != -1 or race_result['result'].find('VIỆ') != -1 or race_result['result'].find(
                'NA') != -1:
            race_result['result'] = 'VIỆT NAM'
        if race_result['result'].strip() == 'VIỆT NAM':
            res.update({
                'nationality': race_result['result'].strip(),
                'nationality_prob': race_result['prob'] if race_result['result'] else '',
                'nationality_prob_old': race_result['prob_old_style'] if race_result['result'] else '',
                'type_new': 'cccd_12_front'
            })
        else:
            res.update({
                'ethnicity': race_result['result'].strip(),
                'ethnicity_prob': race_result['prob'] if race_result['result'] else '',
                'ethnicity_prob_old': race_result['prob_old_style'] if race_result['result'] else '',
                'type_new': 'cmnd_12_front'
            })

        home_result = post_processing.post_process_home_new(img,
                                                            img_name,
                                                            home_coords,
                                                            reader_name=ReaderNew.ADD,
                                                            tf_serving=tf_serving,
                                                            save_dir=path_debug)
        res.update({
            'home': home_result['result'],
            'home_prob': home_result['prob'] if home_result['result'] else '',
            'home_prob_old': home_result['prob_old_style'] if home_result['result'] else ''
        })

        doe_result = {
            'result': ''
        }
        # print(doe_coords)
        doe_coords_tmp = doe_coords.copy()
        doe_coords = []
        for i in range(len(doe_coords_tmp)):
            if doe_coords_tmp[i][0][1] < 375 or doe_coords_tmp[i][0][0] > 450:
                continue
            else:
                doe_coords.append(doe_coords_tmp[i])

        for i in range(len(doe_coords)):
            result = reader.read(tf_serving,
                                 reader_name=ReaderNew.DOB,
                                 img=img,
                                 filename=img_name,
                                 box_coord=doe_coords[i],
                                 save_dir=path_debug)

            ###
            # Check KHONG THOI HAN
            if result['result'] in ['KHÔNG', 'THỜI', 'HẠN', 'KHONG', 'THƠI', 'THOI', 'HAN', 'HẠNG', 'HANG']:
                doe_result = {'result': 'KHÔNG THỜI HẠN', 'prob': 0.9357, 'prob_old_style': 0.3995}
                break
            ###
            tmp_doe = ''
            # print(result)
            for letter in result['result']:
                if not (letter.isalpha()):
                    tmp_doe += letter
            result['result'] = tmp_doe
            if result['result'] != '':
                doe_result = result

        # Re-scaling the prob
        if doe_result['result']:
            prob_doe = doe_result['prob'] + 0.05
            while prob_doe > 1.0:
                prob_doe = prob_doe - 0.05
            doe_result['prob'] = prob_doe

        res.update({
            'doe': doe_result['result'],
            'doe_prob': doe_result['prob'] if doe_result['result'] else '',
            'doe_prob_old': doe_result['prob_old_style'] if doe_result['result'] else ''
        })

        # Remove boxes of "KHÔNG", "THỜI" or "HẠN" from address boxes
        if len(doe_coords) > 1 and doe_result['result'] == 'KHÔNG THỜI HẠN':
            doe_coords_tmp = sorted(doe_coords, key=lambda box: box[0][0])
            max_doe = (doe_coords_tmp[-1][0][0] + doe_coords_tmp[-1][1][0]) / 2
            address_coords_orig = address_coords[:]
            address_coords = []
            for i in range(len(address_coords_orig) - 1):
                if address_coords_orig[i][0][0] > max_doe:
                    address_coords.append(address_coords_orig[i])

        address_result = post_processing.post_process_address_new(img,
                                                                  img_name,
                                                                  address_coords,
                                                                  reader_name=ReaderNew.ADD,
                                                                  tf_serving=tf_serving,
                                                                  save_dir=path_debug)
        res.update({
            'address': address_result['result'],
            'address_prob': address_result['prob'] if address_result['result'] else '',
            'address_prob_old': address_result['prob_old_style'] if address_result['result'] else ''
        })

        # Extract address entities
        address_entities = address_standardization.extract_address_entity(address_result['result'].replace(',', ''))
        res.update({
            'address_entities': address_entities
        })

        # Correct ID
        low_sex_score = 0.89
        up_sex_score = 0.90
        low_dob_score = 0.30
        up_dob_score = 0.35
        low_id_score = 0.29
        up_id_score = 0.30
        if res['sex_prob'] == '':
            sex_score = 0
        else:
            sex_score = prob_sex_old_style
        if res['dob_prob'] == '':
            dob_score = 0
        else:
            dob_score = dob_result['prob_old_style']
        if res['id_prob'] == '':
            id_score = 0
        else:
            id_score = id_result['prob_old_style']
        # print('OLD: ', res['sex'], sex_score)
        # print('OLD: ', res['dob'], dob_score)
        # print('OLD: ', res['id'], id_score)
        res['sex'], res['dob'], res['id'] = id_correction.correction_new_id_with_sex_and_dob(
            res['sex'], sex_score, low_sex_score, up_sex_score,
            res['dob'], dob_score, low_dob_score, up_dob_score,
            res['id'], id_score, low_id_score, up_id_score)

        # print('NEW: ', res['sex'])
        # print('NEW: ', res['dob'])
        # print('NEW: ', res['id'])

    # format probability
    for k in res.keys():
        if (k.endswith('prob') or k.endswith('prob_old')) and res[k] != 'N/A' and res[k] != '':
            res[k] = str('%.2f' % (float(res[k]) * 100))

    if DEBUG_FLAG:
        toc_reader = time.time()
        logging.info('{:25s}'.format("Finish READING in: ") + str(toc_reader - tic_reader) + " s")

    if ENABLE_DRAW:
        cv2.imwrite('detector.jpg', img)

    return res


def back(id_type, img, img_name, tf_serving, issue_loc_new_back):
    """
    :param img:
    :param img_name:
    :param tf_serving:
    :return:
    """
    if DEBUG_FLAG:
        tic_detector = time.time()

    # Detecting
    if id_type == 'old':
        detected_all_fields = detector.detect(tf_serving, 'all-fields-old-back-detector', img)
        batch_size = len(detected_all_fields['scores'])
    else:  # new
        detected_all_fields = detector.detect(tf_serving, 'all-fields-new-back-detector', img)
        batch_size = len(detected_all_fields['scores'])

    if DEBUG_FLAG:
        toc_detector = time.time()
        logging.info('{:25s}'.format("Finish DETECTING in: ") + str(toc_detector - tic_detector) + " s")

    if DEBUG_FLAG:
        tic_reader = time.time()

    eth_coords = list()
    religion_coords = list()
    features_coords = list()
    date_coords = list()
    loc_coords = list()

    # Collect detected boxes
    for i in range(batch_size):
        for j in range(100):

            # COMMENT OUT FEATURE INFORMATION
            if (id_type == 'new' and
                detected_all_fields['classes'][i][j] == 1 and
                detected_all_fields['scores'][i][j] >= 0.3) or (id_type == 'old' and
                                                                detected_all_fields['classes'][i][j] == 3 and
                                                                detected_all_fields['scores'][i][j] >= 0.3):  # features

                features_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                if ENABLE_DRAW and len(features_coords) > 0:
                    img = draw_bounding_box_on_image(img, features_coords[-1][0][1],
                                                     features_coords[-1][0][0],
                                                     features_coords[-1][1][1],
                                                     features_coords[-1][1][0],
                                                     class_type=1,
                                                     use_normalized_coordinates=False)
                continue

            if (id_type == 'new' and
                detected_all_fields['classes'][i][j] == 2 and
                detected_all_fields['scores'][i][j] >= 0.3) or (id_type == 'old' and
                                                                detected_all_fields['classes'][i][j] == 4 and
                                                                detected_all_fields['scores'][i][j] >= 0.35):  # date

                date_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields['boxes'][i][j]))
                if ENABLE_DRAW and len(date_coords) > 0:
                    img = draw_bounding_box_on_image(img, date_coords[-1][0][1],
                                                     date_coords[-1][0][0],
                                                     date_coords[-1][1][1],
                                                     date_coords[-1][1][0],
                                                     class_type=2,
                                                     use_normalized_coordinates=False)
                continue

            if detected_all_fields['classes'][i][j] == 1 and detected_all_fields['scores'][i][j] >= 0.4:  # ethnicity
                eth_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields["boxes"][i][j]))

                if ENABLE_DRAW and len(eth_coords) > 0:
                    img = draw_bounding_box_on_image(img, eth_coords[-1][0][1],
                                                     eth_coords[-1][0][0],
                                                     eth_coords[-1][1][1],
                                                     eth_coords[-1][1][0],
                                                     class_type=3,
                                                     use_normalized_coordinates=False)
                continue

            if detected_all_fields['classes'][i][j] == 2 and detected_all_fields['scores'][i][j] >= 0.4:  # religion
                religion_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields["boxes"][i][j]))

                if ENABLE_DRAW and len(religion_coords) > 0:
                    img = draw_bounding_box_on_image(img, religion_coords[-1][0][1],
                                                     religion_coords[-1][0][0],
                                                     religion_coords[-1][1][1],
                                                     religion_coords[-1][1][0],
                                                     class_type=4,
                                                     use_normalized_coordinates=False)
                continue

            if detected_all_fields['classes'][i][j] == 5 and detected_all_fields['scores'][i][j] >= 0.4:  # issue_loc
                loc_coords.append(
                    reader.convert_bbox_coordinates(img, detected_all_fields["boxes"][i][j]))

                if ENABLE_DRAW and len(loc_coords) > 0:
                    img = draw_bounding_box_on_image(img, loc_coords[-1][0][1],
                                                     loc_coords[-1][0][0],
                                                     loc_coords[-1][1][1],
                                                     loc_coords[-1][1][0],
                                                     class_type=6,
                                                     use_normalized_coordinates=False)
                continue

    # Reading
    res = {}
    # Address reader post-processing ###
    if id_type == 'new':
        res.update({
            'ethnicity': 'N/A',
            'ethnicity_prob': 'N/A',
            'ethnicity_prob_old': 'N/A',
            'religion': 'N/A',
            'religion_prob': 'N/A',
            'religion_prob_old': 'N/A'
        })

        # COMMENT OUT FEATURE INFORMATION
        features_result = post_processing.post_process_feature(img,
                                                               img_name,
                                                               features_coords,
                                                               reader_name=ReaderNew.FEATURE,
                                                               tf_serving=tf_serving,
                                                               save_dir=path_debug,
                                                               card_type='new')

        res.update({
            'features': features_result['result'].strip(),
            'features_prob': features_result['prob'] if features_result['result'] else '',
            'features_prob_old': features_result['prob_old_style'] if features_result['result'] else ''
        })

        date_coords_result = post_processing.post_process_single_line(img,
                                                                      img_name,
                                                                      date_coords,
                                                                      reader_name=ReaderNew.ISSUE_DATE,
                                                                      tf_serving=tf_serving,
                                                                      save_dir=path_debug)

        if date_coords_result['result'].find('L') != -1:
            date_coords_result['result'] = date_coords_result['result'].replace('L', '1')
        if date_coords_result['result'].find('O') != -1:
            date_coords_result['result'] = date_coords_result['result'].replace('O', '0')

        tmp_date = ''
        for letter in date_coords_result['result']:
            if not (letter.isalpha()):
                tmp_date += letter
        date_coords_result['result'] = tmp_date

        res.update({
            'issue_date': date_coords_result['result'].strip().replace(' ', '/'),
            'issue_date_prob': date_coords_result['prob'] if date_coords_result['result'] else '',
            'issue_date_prob_old': date_coords_result['prob_old_style'] if date_coords_result['result'] else '',
            'type_new': 'new_back'
        })

        if issue_loc_new_back == '1':
            # get issue_loc
            try:
                issue_loc_rs = issue_loc.process(tf_serving, 'name2-reader', img)
                if issue_loc_rs != None:
                    res.update({
                        'issue_loc': issue_loc_rs['result'],
                        'issue_loc_prob': issue_loc_rs['prob'],
                        'issue_loc_prob_old': issue_loc_rs['prob_old_style']
                    })
            except:
                logging.info(traceback.format_exc())
                res.update({
                    'issue_loc': 'N/A',
                    'issue_loc_prob': 'N/A',
                    'issue_loc_prob_old': 'N/A'
                })

    else:  # old
        eth_result = post_processing.post_process_single_line(img,
                                                              img_name,
                                                              eth_coords,
                                                              reader_name=ReaderOld.ADD,
                                                              tf_serving=tf_serving,
                                                              save_dir=path_debug)
        # print(eth_result)       
        try:
            corrected_ethinicity, _ = ethnic_correction.ethnic_correction(eth_result['result'].strip().lower())
        except:
            logging.info(traceback.format_exc())
        # print(corrected_ethinicity)
        res.update({
            'ethnicity': corrected_ethinicity.upper(),
            'ethnicity_prob': eth_result['prob'] if eth_result['result'] else '',
            'ethnicity_prob_old': eth_result['prob_old_style'] if eth_result['result'] else ''
        })

        religion_result = post_processing.post_process_single_line(img,
                                                                   img_name,
                                                                   religion_coords,
                                                                   reader_name=ReaderOld.ADD,
                                                                   tf_serving=tf_serving,
                                                                   save_dir=path_debug)
        # print(religion_result)  
        try:
            corrected_religion, _ = religion_correction.religion_correction(religion_result['result'].strip().lower())
        except:
            logging.info(traceback.format_exc())
        # print(corrected_religion)

        res.update({
            'religion': corrected_religion.upper(),
            'religion_prob': religion_result['prob'] if religion_result['result'] else '',
            'religion_prob_old': religion_result['prob_old_style'] if religion_result['result'] else '',
            'type_new': 'old_back'
        })

        features_result = post_processing.post_process_feature(img,
                                                               img_name,
                                                               features_coords,
                                                               reader_name=ReaderOld.FEATURE,
                                                               tf_serving=tf_serving,
                                                               save_dir=path_debug,
                                                               card_type='old')
        res.update({
            'features': features_result['result'].strip(),
            'features_prob': features_result['prob'] if features_result['result'] else '',
            'features_prob_old': features_result['prob_old_style'] if features_result['result'] else ''
        })

        date_coords_result = post_processing.post_process_single_line(img,
                                                                      img_name,
                                                                      date_coords,
                                                                      reader_name=ReaderOld.ADD,
                                                                      tf_serving=tf_serving,
                                                                      save_dir=path_debug)

        if date_coords_result['result'].find('L') != -1:
            date_coords_result['result'] = date_coords_result['result'].replace('L', '1')
        if date_coords_result['result'].find('O') != -1:
            date_coords_result['result'] = date_coords_result['result'].replace('O', '0')

        tmp_date = ''
        for letter in date_coords_result['result']:
            if not (letter.isalpha()):
                tmp_date += letter
        date_coords_result['result'] = tmp_date

        res.update({
            'issue_date': date_coords_result['result'].strip().replace(' ', '/'),
            'issue_date_prob': date_coords_result['prob'] if date_coords_result['result'] else '',
            'issue_date_prob_old': date_coords_result['prob_old_style'] if date_coords_result['result'] else ''
        })

        loc_result = post_processing.post_process_single_line(img,
                                                              img_name,
                                                              loc_coords,
                                                              reader_name=ReaderOld.ADD,
                                                              tf_serving=tf_serving,
                                                              save_dir=path_debug)

        try:
            # print(loc_result['result'])
            # print(address_correction.province_correction(loc_result['result'].strip().lower())[0].upper())
            res.update({
                'issue_loc': address_correction.province_correction(loc_result['result'].strip().lower())[0].upper(),
                'issue_loc_prob': loc_result['prob'] if loc_result['result'] else '',
                'issue_loc_prob_old': loc_result['prob_old_style'] if loc_result['result'] else ''
            })
        except:
            logging.info(traceback.format_exc())
            res.update({
                'issue_loc': '',
                'issue_loc_prob': '',
                'issue_loc_prob_old': '',
            })

    # format probability
    for k in res.keys():
        if (k.endswith('prob') or k.endswith('prob_old')) and res[k] != 'N/A' and res[k] != '':
            res[k] = str('%.2f' % (float(res[k]) * 100))

    if DEBUG_FLAG:
        toc_reader = time.time()
        logging.info('{:25s}'.format("Finish READING in: ") + str(toc_reader - tic_reader) + " s")

    if ENABLE_DRAW:
        cv2.imwrite('detector.jpg', img)

    return res
