import argparse
import glob
import os

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_df(excel_path):
    """Load Excel as DataFrame and preprocess
    """
    ID_df = pd.read_excel(excel_path)

    # Drop chore columns
    #ID_df.drop(columns=['Unnamed: 6'], inplace=True)

    # Add img_id/image_name column (extracted from MATTRUOCCMND)
    ID_df['img_name'] = ID_df.apply(lambda row: row['MATTRUOCCMND'].split('/')[-1], axis=1)
    ID_df['img_id'] = ID_df.apply(lambda row: row['img_name'].split('.')[0], axis=1)

    return ID_df


def get_bounding_box(pts, margin=None):
    """Get bounding box from polygon
    Args:
        pts: numpy array [[x1,y1], [x2, y2], [x3, y3], [x4, y4]] of polygon
        (x1, y1) ----- (x4, y4)
            |              |
            |              |
        (x2, y2) ----- (x3, y3)
        margin: Increase size of bounding box by margin pixel
    Returns: numpy array [[xmin, ymin], [xmax, ymax]]
    """
    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)

    if margin is None:
        return np.array([[xmin, ymin], [xmax, ymax]])
    else:
        return np.array([[xmin-margin, ymin-margin], [xmax+margin, ymax+margin]])


def in_name_region(pts, ul=[340, 160], rd=[800, 250]):
    """
    Args:
        xmin, ymin, xmax, ymax: bounding box of name region
         ul (xmin, ymin) -----
             |               |
             |               |
             --------- rd (xmax, ymax)
    """
    box = get_bounding_box(pts)
    x_polygon_min, y_polygon_min = box[0]
    x_polygon_max, y_polygon_max = box[1]
    xmin, ymin = ul
    xmax, ymax = rd
    return (x_polygon_min >= xmin) and (y_polygon_min >= ymin) and \
        (x_polygon_max <= xmax) and (y_polygon_max <= ymax)


def main(args):
    """Prepare words training for Word Reader/Detector

    Algorithm:
        1. Traverse all co-ordinates annotation files
        2. For each annotation, read each line. Take all polygons
        (defined by co-ordinates in each line) in name region.
        3. Count number of polygons taken in name region,
        compare with length of name corresponding to img_id
        4. If equal, crop the box (transformed from polygon with larger margin) images
        and apply corresponding label

    Sample usage:
        python data/extract_words.py --dataset_dir dataset/id/EAST/raw \
            --annotation_dir dataset/id/EAST/annotation --excel dataset/id/HINHCMND.xlsx \
            --output_dir dataset/id/EAST/words
    """
    # load excel
    ID_df = load_df(args.excel)

    # create output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for anno in tqdm(glob.iglob(os.path.join(args.annotation_dir, '*.txt'))):
        coords = open(anno).readlines()

        img_id = anno.split('/')[-1].split('.')[0]
        search = ID_df.loc[ID_df['img_id'] == img_id]

        if search.values.any():
            # turn name into list of words
            name = search['Hoten'].values[0].split()
            # get corresponding image path
            img_path = os.path.join(args.dataset_dir, search['img_name'].values[0])

            if os.path.exists(img_path):
                # load img
                img = cv2.imread(img_path)

                # get all polygon in name region
                polygons = []
                for line in coords:
                    coord = list(map(int, line.strip().split(',')))
                    pts = np.array(coord).reshape([-1, 2]) # zip
                    if in_name_region(pts):
                        polygons.append(pts)

                # compare with length of name (number of words)
                if len(polygons) == len(name):
                    # sort polygons by x
                    polygons = np.array(polygons)
                    polygons = np.sort(polygons, axis=0)

                    # save bounding box (with margin) with corresponding word label
                    for k, pts in enumerate(polygons):
                        box = get_bounding_box(pts, margin=12)
                        crop = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]

                        out_file = os.path.join(args.output_dir, name[k]+'.jpg')
                        cv2.imwrite(out_file, crop)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        help='Dataset directory contains raw images')
    parser.add_argument('--annotation_dir', type=str,
                        help='Directory with annotation files containing polygon co-ordinates')
    parser.add_argument('--excel', type=str,
                        help='Excel file containing annotation of all records')
    parser.add_argument('--output_dir', type=str,
                        help='Result directory containing cropped words')

    args = parser.parse_args()
    main(args)
