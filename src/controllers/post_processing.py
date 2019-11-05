import functools
import logging
import re

from src.controllers import reader
from src.others.correct_addr.address_correction import AddressCorrection
import src.others.post_processing_utils as utils


####################################
# Remove Vietnamese accent
INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđ"
INTAB = [ch for ch in INTAB]

OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"

r = re.compile("|".join(INTAB))
replaces_dict = dict(zip(INTAB, OUTTAB))

def remove_accents(utf8_str):
    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)
####################################


address_correction = AddressCorrection()


def calculate_center_coord(list_boxes):
    list_center_coords = list()
    for box in list_boxes:
        list_center_coords.append((box[0][1] + box[1][1]) / 2)
    
    return sum(list_center_coords) / len(list_center_coords)


def post_process_name(cropped_image, filename, name_coords, reader_name, tf_serving, save_dir):
    """
    Post-process to arrange name boxes in the  correct order
    :param cropped_image:
    :param filename:
    :param name_coords:
    :param reader_name:
    :param tf_serving:
    :return: name
    """
    name_coords = sorted(name_coords, key=lambda box: [box[0][1], box[0][0]])
    tmp_coords = name_coords.copy()
    # print(cropped_image.shape)
    for i in tmp_coords:
        if i[0][1] > cropped_image.shape[0]/2:
            name_coords.remove(i)
    # print(name_coords)

    if len(name_coords) > 0:
        name_1 = [name_coords[0]]
    else:
        name_1 = []
    name_2 = []

    for i in range(1, len(name_coords)):
        if name_coords[i][0][1] - name_coords[0][0][1] < 15:
            name_1.append(name_coords[i])
        else:
            name_2.append(name_coords[i])

    #TODO: LAP CODE - CAN REFACTOR
    if len(name_1) > 0 and len(name_2) > 0:
        center_1 = calculate_center_coord(name_1)
        center_2 = calculate_center_coord(name_2)
        if center_2 - center_1 > 65:
            name_coords = name_2
            if len(name_coords) > 0:
                name_1 = [name_coords[0]]
            else:
                name_1 = []
            name_2 = []

            for i in range(1, len(name_coords)):
                if name_coords[i][0][1] - name_coords[0][0][1] < 15:
                    name_1.append(name_coords[i])
                else:
                    name_2.append(name_coords[i])
    ######################################
    # for b in name_1:
    #     utils.draw_rec(cropped_image, b)

    name_1 = utils.get_different_bboxs(name_1)
    name_2 = utils.get_different_bboxs(name_2)

    # sort by x1 for each name line
    name_1 = sorted(name_1, key=lambda box: box[0][0])
    name_2 = sorted(name_2, key=lambda box: box[0][0])

    # for b in name_1:
    #     utils.draw_rec(cropped_image, b)
    # cv2.imwrite('img.jpg', cropped_image)

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='name_{}'.format(filename),
                            box_coords=(name_1 + name_2),
                            save_dir=save_dir)
    # print(res)
                            
    if reader_name == 'name-reader':
        res2 = reader.read_batch(tf_serving,
                            reader_name='name2-reader',
                            img=cropped_image,
                            filename='name_{}'.format(filename),
                            box_coords=(name_1 + name_2),
                            save_dir=save_dir)
        # print(res2)
        for i in range(len(res)):
            if res[i][0].strip()[0] == 'T':
                tmp = res[i][0][1:]
                no_accent_1 = remove_accents(tmp.lower())
                no_accent_2 = remove_accents(res2[i][0].lower())
                if no_accent_1 == no_accent_2:
                    res[i] = (res[i][0][1:], res[i][1], res[i][2])
            if res[i][1] < 0.1 and res2[i][1] > 0.4:
                res[i] = res2[i]
    # print(res)
    for i in range(len(res)):
        if res[i][0].find('0') != -1:
            res[i] = (res[i][0].replace('0', 'O'), res[i][1], res[i][2])
        if res[i][0].find('1') != -1:
            res[i] = (res[i][0].replace('1', 'I'), res[i][1], res[i][2])
        if res[i][0].find('F') != -1:
            res[i] = (res[i][0].replace('F', 'P'), res[i][1], res[i][2])
        if res[i][0].find('LUU') != -1:
            res[i] = (res[i][0].replace('LUU', 'LƯU'), res[i][1], res[i][2])
        if res[i][0].find('HUU') != -1:
            res[i] = (res[i][0].replace('HUU', 'HỮU'), res[i][1], res[i][2])
        if res[i][0].find('LONN') != -1:
            res[i] = (res[i][0].replace('LONN', 'LOAN'), res[i][1], res[i][2])
        if res[i][0].find('LHỊ') != -1:
            res[i] = (res[i][0].replace('LHỊ', 'THỊ'), res[i][1], res[i][2])
        if res[i][0].find(' UU') != -1:
            res[i] = (res[i][0].replace(' UU', 'U'), res[i][1], res[i][2])
        if res[i][0].find('LVĂN') != -1:
            res[i] = (res[i][0].replace('LVĂN', 'VĂN'), res[i][1], res[i][2])
        if res[i][0] == 'LUYỄN':
            res[i] = ('NGUYỄN', res[i][1], res[i][2])

    for i in range(len(res)):
        if not res[i][0].isalpha():
            res[i] = ('', 1.0, 1.0)
    # print(res)
    # extract results
    if len(res) == 0:
        return {
            'result': '',
            'prob': '',
            'prob_old_style': ''
        }
    else:
        # Re-scaling the prob
        prob = functools.reduce(lambda x, y: x * y, [prob for name_part, prob, prob_old in res]) + 0.25
        while prob > 1.0:
            prob = prob - 0.02

        return {
            'result': ' '.join([name_part for name_part, prob, prob_old in res]).strip(),
            'prob': prob,
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for name_part, prob, prob_old in res])
        }


def post_process_address_old(cropped_image, filename, address_coords, reader_name, tf_serving, save_dir):
    """
    Post-process to arrange address boxes in the correct order
    :param cropped_image:
    :param filename:
    :param address_coords:
    :param reader_name:
    :param tf_serving:
    :return: address
    """
    address_coords = sorted(address_coords, key=lambda box: [box[0][1], box[0][0]])
    if len(address_coords) > 0:
        address_2 = [address_coords[-1]]
    else:
        address_2 = []
    address_1 = []

    # Arrange 2 lines of address based on relative distance
    distance_same_line = -22
    distance_diff_line = -70
    home_coords = address_coords[:]

    for i in range(0, len(address_coords) - 1):
        if address_coords[i][0][1] - address_coords[-1][0][1] > distance_same_line:
            address_2.append(address_coords[i])
            home_coords.remove(address_coords[i])  # pop add, so we have home left
        elif address_coords[i][0][1] - address_coords[-1][0][1] > distance_diff_line:
            address_1.append(address_coords[i])
            home_coords.remove(address_coords[i])  # pop add, so we have home left
        else:
            continue

    if len(address_coords) > 0:
        home_coords.remove(address_coords[-1])
    
    # for b in (address_1+address_2):
    #     utils.draw_rec(cropped_image, b)

    address_1 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(address_1))
    address_2 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(address_2))

    # for b in (address_1+address_2):
    #     utils.draw_rec(cropped_image, b)
    # cv2.imwrite('img_a.jpg', cropped_image)

    # sort by x1 for each address line
    address_1 = sorted(address_1, key=lambda box: box[0][0])
    address_2 = sorted(address_2, key=lambda box: box[0][0])

    # print(address_1)
    # print(address_2)

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='address_{}'.format(filename),
                            box_coords=(address_1 + address_2),
                            save_dir=save_dir)

    ##############
    # for bbox in (address_2):
    # utils.draw_rec(cropped_image, address_2[0])
    # cv2.imwrite('img.jpg' ,cropped_image)
    ##############

    # Result dict
    address = {'result': '', 'prob': '', 'prob_old_style': ''}
    if len(res) > 0:
        address = {
            'result': ' '.join([addr_part for addr_part, prob, prob_old in res]).strip(),
            'prob': functools.reduce(lambda x, y: x * y, [prob for addr_part, prob, prob_old in res]),
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for addr_part, prob, prob_old in res])
        }

    ###################################################
    # Home
    if len(home_coords) > 0:
        home_2 = [home_coords[-1]]
    else:
        home_2 = []
    home_1 = []

    for i in range(0, len(home_coords) - 1):
        if home_coords[i][0][1] > 280:
            if home_coords[i][0][1] - home_coords[-1][0][1] > distance_same_line:
                home_2.append(home_coords[i])
            elif home_coords[i][0][1] - home_coords[-1][0][1] > distance_diff_line:
                home_1.append(home_coords[i])
            else:
                continue

    home_1 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(home_1))
    home_2 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(home_2))

    # sort by x1 for each address line
    home_1 = sorted(home_1, key=lambda box: box[0][0])
    home_2 = sorted(home_2, key=lambda box: box[0][0])

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='home_{}'.format(filename),
                            box_coords=(home_1 + home_2),
                            save_dir=save_dir)

    # Result dict
    home = {'result': '', 'prob': '', 'prob_old_style': ''}
    if len(res) > 0:
        home = {
            'result': ' '.join([home_part for home_part, prob, prob_old in res]).strip(),
            'prob': functools.reduce(lambda x, y: x * y, [prob for home_part, prob, prob_old in res]),
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for home_part, prob, prob_old in res])
        }

    # Correct addresses
    # print('HOME:', home['result'])
    # print('ADDRESS:', address['result'])
    if address['result'].find('T7THỂ') != -1:
        address['result'] = address['result'].replace('T7THỂ', 'T/THỂ')
    if address['result'].find('VKSDTC') != -1:
        address['result'] = address['result'].replace('VKSDTC', 'VKSNDTC')
    if address['result'].find('|') != -1:
        address['result'] = address['result'].replace('|', '-')
    if address['result'].find('7P') != -1:
        address['result'] = address['result'].replace('7P', 'TP')
    if address['result'].find(' B.M.1 ') != -1:
        address['result'] = address['result'].replace(' B.M.1 ', ' B.M.T ')
    if address['result'].find(' MI1 ') != -1:
        address['result'] = address['result'].replace(' MI1 ', ' MIL ')
    try:
        if address['result'].split(' ')[-2] == 'MINH':
            address['result'] = address['result'][:-1]
    except:
        pass
    corrected_add = str()
    if address['result'].find('.') != -1:
        corrected_add = address['result'].replace('.', ' ')
    else:
        corrected_add = address['result']
    corrected_add = address_correction.address_correction(corrected_add.lower())[0].upper()
    # print(corrected_add)

    if corrected_add.replace(',', '') != address['result']:
        if address['prob_old_style']*100 < 20.0:
            address['result'] = corrected_add
    else:
        address['result'] = corrected_add
    
    corrected_home = str()
    if home['result'].find('.') != -1:
        corrected_home = home['result'].replace('.', ' ')
    else:
        corrected_home = home['result']
    corrected_home = address_correction.address_correction(corrected_home.lower())[0].upper()
    # print(corrected_home)
    if corrected_home.replace(',', '') != home['result']:
        if home['prob_old_style']*100 < 20.0:
            home['result'] = corrected_home
    else:
        home['result'] = corrected_home

    # if address['result'].find(',') != -1:
    #     address['prob'] += 0.1
    # if home['result'].find(',') != -1:
    #     home['prob'] += 0.1

    # Re-scaling the prob
    if address['prob'] != '':
        prob_address = address['prob'] + 0.50
        while prob_address > 1.0:
            prob_address = prob_address - 0.05
        address['prob'] = prob_address

    if home['prob'] != '':
        prob_home = home['prob'] + 0.50
        while prob_home > 1.0:
            prob_home = prob_home - 0.05
        home['prob'] = prob_home

    return address, home


def post_process_single_line(cropped_image, filename, coords, reader_name, tf_serving, save_dir):
    """

    :param cropped_image:
    :param filename:
    :param race_coords:
    :param reader_name:
    :param tf_serving:
    :param save_dir:
    """
    coords = sorted(coords, key=lambda box: box[0][0])

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='single_line_{}'.format(filename),
                            box_coords=coords,
                            save_dir=save_dir)
    
    # Re-scaling the prob
    comp = 0
    if len(coords) == 1:
        comp = 0.1
    elif len(coords) == 2:
        comp = 0.17
    elif len(coords) == 3:
        comp = 0.23
    else:
        comp = 0.30

    # Result dict
    race = {'result': '', 'prob': '', 'prob_old_style': ''}
    if len(res) > 0:
        prob = functools.reduce(lambda x, y: x * y, [prob for addr_part, prob, prob_old in res]) + comp
        while prob > 1.0:
            prob = prob - 0.05
        race = {
            'result': ' '.join([race_part for race_part, prob, prob_old in res]).strip(),
            'prob': prob,
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for addr_part, prob, prob_old in res])
        }

    return race


def post_process_address_new(cropped_image, filename, address_coords, reader_name, tf_serving, save_dir):
    """
    Post-process to arrange address boxes in the correct order
    :param cropped_image:
    :param filename:
    :param address_coords:
    :param reader_name:
    :param tf_serving:
    :return: address
    """
    address_coords = sorted(address_coords, key=lambda box: [box[0][1], box[0][0]])
    if len(address_coords) > 0:
        address_2 = [address_coords[-1]]
    else:
        address_2 = []
    address_1 = []

    # Arrange 2 lines of address based on relative distance
    distance_same_line = -15
    distance_diff_line = -70

    for i in range(0, len(address_coords) - 1):
        if address_coords[i][0][1] - address_coords[-1][0][1] > distance_same_line:
            address_2.append(address_coords[i])
        elif address_coords[i][0][1] - address_coords[-1][0][1] > distance_diff_line:
            address_1.append(address_coords[i])
        else:
            continue

    address_1 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(address_1))
    address_2 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(address_2))

    # sort by x1 for each address line
    address_1 = sorted(address_1, key=lambda box: box[0][0])
    address_2 = sorted(address_2, key=lambda box: box[0][0])

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='address_{}'.format(filename),
                            box_coords=(address_1 + address_2),
                            save_dir=save_dir)

    # Result dict
    address = {'result': '', 'prob': '', 'prob_old_style': ''}
    if len(res) > 0:
        # Re-scaling the prob
        prob = functools.reduce(lambda x, y: x * y, [prob for addr_part, prob, prob_old in res]) + 0.50
        while prob > 1.0:
            prob = prob - 0.05
        address = {
            'result': ' '.join([addr_part for addr_part, prob, prob_old in res]).strip(),
            'prob': prob,
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for addr_part, prob, prob_old in res])
        }

    # Correct address
    # print('ADDRESS:', address['result'])
    address['result'] = address_correction.address_correction(address['result'].lower())[0].upper()
    # print(address['result'])

    # if address['result'].find(',') != -1:
    #     address['prob'] += 0.1

    return address


def post_process_home_new(cropped_image, filename, home_coords, reader_name, tf_serving, save_dir):
    """
    Post-process to arrange address boxes in the correct order
    :param cropped_image:
    :param filename:
    :param home_coords:
    :param reader_name:
    :param tf_serving:
    :return: address
    """
    home_coords = sorted(home_coords, key=lambda box: [box[0][1], box[0][0]])
    if len(home_coords) > 0:
        home_1 = [home_coords[0]]
    else:
        home_1 = []
    home_2 = []

    # Arrange 2 lines of address based on relative distance
    distance_same_line = 15
    distance_diff_line = 60

    for i in range(1, len(home_coords)):
        if home_coords[i][0][1] - home_coords[0][0][1] < distance_same_line:
            home_1.append(home_coords[i])
        elif home_coords[i][0][1] - home_coords[0][0][1] < distance_diff_line:
            home_2.append(home_coords[i])
        else:
            continue

    home_1 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(home_1))
    home_2 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(home_2))

    # sort by x1 for each address line
    home_1 = sorted(home_1, key=lambda box: box[0][0])
    home_2 = sorted(home_2, key=lambda box: box[0][0])

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='home_{}'.format(filename),
                            box_coords=(home_1 + home_2),
                            save_dir=save_dir)

    # Result dict
    home = {'result': '', 'prob': '', 'prob_old_style': ''}
    if len(res) > 0:
        # Re-scaling the prob
        prob = functools.reduce(lambda x, y: x * y, [prob for addr_part, prob, prob_old in res]) + 0.50
        while prob > 1.0:
            prob = prob - 0.05
        home = {
            'result': ' '.join([home_part for home_part, prob, prob_old in res]).strip(),
            'prob': prob,
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for addr_part, prob, prob_old in res])
        }

    # Correct address
    home['result'] = address_correction.address_correction(home['result'].lower())[0].upper()

    # if home['result'].find(',') != -1:
    #     home['prob'] += 0.1

    return home


def post_process_feature(cropped_image, filename, feature_coords, reader_name, tf_serving, save_dir, card_type):
    """
    Post-process to arrange address boxes in the correct order
    :param cropped_image:
    :param filename:
    :param home_coords:
    :param reader_name:
    :param tf_serving:
    :return: address
    """
    feature_coords = sorted(feature_coords, key=lambda box: [box[0][1], box[0][0]])
    tmp_coords = feature_coords.copy()
    # print(cropped_image.shape)
    if card_type == 'old':
        for i in tmp_coords:
            if i[0][1] < 100:
                feature_coords.remove(i)
            elif i[0][1] > cropped_image.shape[0]/2:
                feature_coords.remove(i)

    if len(feature_coords) > 0:
        feat_1 = [feature_coords[0]]
    else:
        feat_1 = []
    feat_2 = []
    feat_3 = []

    # Arrange 2 lines of address based on relative distance
    if card_type == 'old':
        distance_same_line = 20 
        distance_diff_line = 65
    else:
        distance_same_line = 10
        distance_diff_line = 80

    for i in range(1, len(feature_coords)):
        if feature_coords[i][0][1] - feature_coords[0][0][1] < distance_same_line:
            feat_1.append(feature_coords[i])
        elif feature_coords[i][0][1] - feature_coords[0][0][1] < distance_diff_line:
            feat_2.append(feature_coords[i])
        elif card_type == 'old':
            feat_3.append(feature_coords[i])
        else:
            continue

    feat_1 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(feat_1))
    feat_2 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(feat_2))
    feat_3 = utils.remove_intersecting_bbox(utils.merge_bbox_in_vertical_align(feat_3))

    # sort by x1 for each address line
    feat_1 = sorted(feat_1, key=lambda box: box[0][0])
    feat_2 = sorted(feat_2, key=lambda box: box[0][0])
    feat_3 = sorted(feat_3, key=lambda box: box[0][0])

    res = reader.read_batch(tf_serving,
                            reader_name=reader_name,
                            img=cropped_image,
                            filename='home_{}'.format(filename),
                            box_coords=(feat_1 + feat_2 + feat_3),
                            save_dir=save_dir)

    # Result dict
    features = {'result': '', 'prob': '', 'prob_old_style': ''}
    if len(res) > 0:
        # Re-scaling the prob
        prob = functools.reduce(lambda x, y: x * y, [prob for feat_part, prob, prob_old in res]) + 0.35
        while prob > 1.0:
            prob = prob - 0.05
        features = {
            'result': ' '.join([feat_part for feat_part, prob, prob_old in res]).strip(),
            'prob': prob,
            'prob_old_style': functools.reduce(lambda x, y: x * y, [prob_old for feat_part, prob, prob_old in res])
        }

    return features

