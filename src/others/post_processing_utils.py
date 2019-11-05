import cv2
import traceback

def merge_bbox_in_vertical_align(bboxs):
    bboxs1 = list(bboxs)
    bboxs2 = list(bboxs1)

    try:
        for R1 in bboxs:
            for R2 in bboxs1:
                if check_in_vertical_align(R1, R2):
                    area_min = min(calculate_area(R1), calculate_area(R2))
                    if (R1[0][0] == R2[0][0] and R1[0][1] ==  R2[0][1] and R1[1][0]== R2[1][0] and R1[1][1]== R2[1][1]) or \
                            (calculate_area_intersecting(R1, R2) < area_min / 3):
                        continue
                    else:
                        R3 = [[min(R1[0][0], R2[0][0]), min(R1[0][1], R2[0][1])],
                            [max(R1[1][0], R2[1][0]), max(R1[1][1], R2[1][1])]]
                        if R1 in bboxs2:
                            bboxs2.remove(R1)
                        bboxs2.remove(R2)
                        bboxs2.append(R3)

            bboxs1 = list(bboxs2)
            bboxs1 = list(remove_duplicate(bboxs1))
        bboxs = list(bboxs1)
    except:
        logging.error(traceback.format_exc())
        return bboxs

    return bboxs

def remove_duplicate(duplicate):
    final_list = []
    for num in duplicate:
        if num not in final_list:
            final_list.append(num)
    return final_list

def check_in_vertical_align(R1,R2):
    if (R1[0][0] >= R2[1][0]) or (R1[1][0] <= R2[0][0]):
        return False
    else:
        return True

def remove_intersecting_bbox(bboxs):
    bboxs1 = list(bboxs)
    bboxs2 = list(bboxs1)
    for R1 in bboxs:
        for R2 in bboxs1:
            if check_overlap(R1, R2):
                i_a = calculate_area_intersecting(R1, R2)
                R1_a = calculate_area(R1)
                R2_a = calculate_area(R2)
                area_min1 = min(R1_a, R2_a)
                if  i_a > area_min1 / 2  and not (R1[0][0] == R2[0][0] and R1[0][1] ==  R2[0][1] and R1[1][0]== R2[1][0] and R1[1][1]== R2[1][1]):

                    if R1_a > R2_a:
                        if R2 in bboxs2:
                            bboxs2.remove(R2)
                    elif R1 in bboxs2:
                        bboxs2.remove(R1)

        bboxs1 = list(bboxs2)
    bboxs = list(bboxs1)
    return bboxs


# def draw_rec(img, bbox):
#     cv2.rectangle(img,(bbox[0][0],bbox[0][1]),(bbox[1][0],bbox[1][1]),0,1) # draw bounding box in summary image
#     return img


def get_bbox(box):
    (wordBox,wordImg) = box
    (x, y, w, h) = wordBox
    return [[x,y],[x + w,y + h]]

def check_overlap(R1,R2):
    if (R1[0][0] > R2[1][0]) or (R1[1][0] < R2[0][0]) or \
            (R1[0][1] > R2[1][1]) or (R1[1][1] < R2[0][1]):
        return False
    else:
        return True

def calculate_area(R1):
    return abs(R1[1][0] - R1[0][0])*abs(R1[1][1] - R1[0][1])

def calculate_area_intersecting(R1, R2):  # returns None if rectangles don't intersect
    dx = min(R1[1][0], R2[1][0]) - max(R1[0][0], R2[0][0])
    dy = min(R1[1][1], R2[1][1]) - max(R1[0][1], R2[0][1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def draw_rec(img, bbox):
    cv2.rectangle(img,(bbox[0][0],bbox[0][1]),(bbox[1][0],bbox[1][1]),(255,0,0), 1) # draw bounding box in summary image


def get_different_bboxs(bboxs):

    bboxs1 = list(bboxs)
    bboxs2 = list(bboxs1)

    try:
        for R1 in bboxs:
            for R2 in bboxs1:

                if (R1[0][0] == R2[0][0] and R1[0][1] ==  R2[0][1] and R1[1][0]== R2[1][0] and R1[1][1]== R2[1][1]):
                    continue

                if check_overlap(R1, R2):
                    i_a = calculate_area_intersecting(R1, R2)
                    R1_a = calculate_area(R1)
                    R2_a = calculate_area(R2)
                    area_min = min(R1_a, R2_a)
                    area_max = max(R1_a,R2_a)
        
                    if i_a > area_min*0.7 and  i_a < area_max*0.6 :
                        if R1_a > R2_a :
                            C_x = R1[0][0] + int((R1[1][0] - R1[0][0])/2)
                            R3 = [[R1[0][0],R1[0][1]],[C_x,R1[1][1]]]
                            R4 = [[C_x,R1[0][1]],[R1[1][0],R1[1][1]]]

                            if calculate_area_intersecting(R2,R3) > calculate_area_intersecting(R2,R4):
                                R_a = [[min(R1[0][0],R2[0][0]),min(R1[0][1],R2[0][1])],[R2[1][0],max(R2[1][1],R1[1][1])]]
                                R_b = [[R2[1][0],R1[0][1]],[R1[1][0],R1[1][1]]]
    
                            else:
                                R_a = [[R1[0][0],R1[0][1]],[R2[0][0],R1[1][1]]]
                                R_b = [[R2[0][0],min(R1[0][1],R2[0][1])],[max(R1[1][0],R2[1][0]),max(R1[1][1],R2[1][1])]]
                        else:
                            C_x = R2[0][0] + int((R2[1][0] - R2[0][0])/2)
                            R3 = [[R2[0][0],R2[0][1]],[C_x,R2[1][1]]]
                            R4 = [[C_x,R2[0][1]],[R2[1][0],R2[1][1]]]

                            if calculate_area_intersecting(R1,R3) > calculate_area_intersecting(R1,R4):
                                R_a = [[min(R2[0][0],R1[0][0]),min(R2[0][1],R1[0][1])],[R1[1][0],max(R2[1][1],R1[1][1])]]
                                R_b = [[R1[1][0],R2[0][1]],[R2[1][0],R2[1][1]]]
                            else:
                                R_a = [[R2[0][0],R2[0][1]],[R1[0][0],R2[1][1]]]
                                R_b = [[R1[0][0],min(R2[0][1],R1[0][1])],[max(R2[1][0],R1[1][0]),max(R2[1][1],R1[1][1])]]

                        if R1 in bboxs2:
                            bboxs2.remove(R1)
                        if R2 in bboxs2:
                            bboxs2.remove(R2)

                        bboxs2.append(R_a)
                        bboxs2.append(R_b)

            bboxs1 = list(bboxs2)
            bboxs1 = list(remove_duplicate(bboxs1))
            bboxs1 = remove_intersecting_bbox(bboxs1)
        bboxs = list(bboxs1)
    except:
        print(traceback.format_exc())
        return bboxs
    return bboxs

