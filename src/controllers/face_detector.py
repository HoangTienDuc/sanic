import numpy as np
import imutils

from src.models.face_detector import mtcnn


def crop_face(img, crop_size=160, crop_margin=10, threshold=0.98, tf_serving='localhost:8500'):
    """Detect face using MTCNN and then resize image
    
    Args:
    threshold: probability minimum to define if a box contained face
    """
    
    # Keep rotating image until detecting face
    # with probability > threshold

    boxes, _ = mtcnn.detect_face(img, tf_serving=tf_serving)
    face_boxes = list()

    for index, bbox in enumerate(boxes):
        if bbox[4] < threshold:
            continue
        else:
            face_boxes.append(bbox)
    
    # Filter skewed
    # depends on MTCNN landmarks

    # Filter faces on background/poster
    # build a classifier (?)
    if len(face_boxes) == 0:
        return None

    # Get max accuracy
    best_bbox = max(face_boxes, key=lambda bbox: bbox[4])

    # Crop & Resize image
    h, w = img.shape[:2]

    left = int(np.maximum(best_bbox[0] - crop_margin / 2, 0))
    top = int(np.maximum(best_bbox[1] - crop_margin / 2, 0))
    right = int(np.minimum(best_bbox[2] + crop_margin / 2, w))
    bottom = int(np.minimum(best_bbox[3] + crop_margin / 2, h))
    
    crop_img = img[top:bottom, left:right, :]
    # crop_img = cv2.resize(crop_img, (crop_size, crop_size),
    #                       interpolation=cv2.INTER_LINEAR)

    return crop_img
