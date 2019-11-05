import cv2
import traceback
import time
import logging
from src.common import DEBUG_FLAG
from src.models.craft import detector_craft
from src.controllers import post_processing

TYPE1 = 'CỤC TRƯỞNG CỤC CẢNH SÁT ĐKQL CƯ TRÚ VÀ DLQG VỀ DÂN CƯ'
TYPE2 = 'CỤC TRƯỞNG CỤC CẢNH SÁT QUẢN LÝ HÀNH CHÍNH VỀ TRẬT TỰ XÃ HỘI'
TYPE3 = 'CỤC TRƯỞNG CỤC CSQLHC VỀ TTXH'


craft_model = detector_craft.load_model()


def process(tf_serving, reader_name, crop_img):
    try:
        # detecting
        if DEBUG_FLAG:
            tic_detector = time.time()

        detected_bboxes = detector_craft.detect(crop_img, craft_model)

        if DEBUG_FLAG:
            toc_detector = time.time()
            logging.info('{:25s}'.format("Finish DETECTING in: ") + str(toc_detector - tic_detector) + " s")

        # preprocess boxs
        boxs_rs= preprocess_boxs(detected_bboxes)

        # if boxs_rs is not None:
        #     for box in list(boxs_rs):
        #         bottom_right = (box[1][0], box[1][1])
        #         top_left = (box[0][0], box[0][1])
        #         cv2.rectangle(crop_img, top_left, bottom_right, (255,0,0), 2)
        # cv2.imwrite('rs.jpg',crop_img)

        if boxs_rs is None or len(boxs_rs) < 0:
            return None

        # process
        rs_date = post_processing.post_process_single_line(crop_img,'', boxs_rs, reader_name, tf_serving,'')
        #print(rs_date)

        result = set(str(rs_date['result']).split())
        a_set = set(['ĐKQL', 'DLQG'])
        b_set = set('QUẢN LÝ HÀNH CHÍNH TRẬT TỰ XÃ HỘI'.split())
        c_set = set(['CSQLHC', 'TTXH'])

        # check type
        if len(result.intersection(a_set)) > 0:
            rs_date.update(
                {
                    'result': TYPE1
                }
            )
        elif len(result.intersection(b_set)) > 0:
            rs_date.update(
                {
                    'result': TYPE2
                }
            )
        elif len(result.intersection(c_set)) > 0:
            rs_date.update(
                {
                    'result': TYPE3
                }
            )
        else:
            rs_date.update(
                {
                    'result': TYPE1
                }
            )

        return rs_date
    except:
        logging.exception(traceback.format_exc())
        return None


def preprocess_boxs(boxs):
    try:
        if boxs is None or len(boxs) < 1:
            return None, None

        boxs_rs = []

        for box in boxs:
            if box[0][0] > 420 and box[0][1] > 240 and box[0][1] < 350:
                boxs_rs.append(box)

        if len(boxs_rs) < 1 :
            return None

        return boxs_rs
    except:
        logging.error(traceback.format_exc())
    return None


if __name__ == '__main__':
    image = cv2.imread('/home/green/Pictures/2019-09-06.jpg')
    image = cv2.resize(image, (800,500))
    process('', '', image)
