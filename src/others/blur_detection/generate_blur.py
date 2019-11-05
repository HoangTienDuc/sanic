import argparse
import os
import cv2
from tqdm import tqdm
from .blur_generate import BoxBlur, DefocusBlur

def generate_fn(st_i, existed_fn, prefix=None):
    while True:
        st_i += 1
        blur_fn = 'blur_{}.jpg'.format(st_i)
        if blur_fn not in existed_fn:
            if prefix is None:
                return blur_fn, st_i
            else:
                return os.path.join(prefix, blur_fn), st_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='Directory contains images')
    parser.add_argument('--dist', help='Directory contains blurred')
    parser.add_argument('--nb_of_image', help='number of image generate', default=100)
    args = parser.parse_args()
    src_path = args.src
    dist_path = args.dist
    nb_of_image = int(args.nb_of_image)
    img_fns = os.listdir(src_path)
    dist_fns = set(os.listdir(dist_path))
    i = 0
    c = 0
    progress_bar = tqdm(total=nb_of_image)
    for img_fn in img_fns:
        img = cv2.imread(os.path.join(src_path, img_fn), cv2.IMREAD_GRAYSCALE)
        box_blur = BoxBlur(img, 9)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        box_blur = BoxBlur(img, 11)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        box_blur = BoxBlur(img, 13)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        box_blur = BoxBlur(img, 15)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)

        defocus_blurred = DefocusBlur(img, 9)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        defocus_blurred = DefocusBlur(img, 11)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        defocus_blurred = DefocusBlur(img, 13)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        defocus_blurred = DefocusBlur(img, 15)
        new_fn, i = generate_fn(i, dist_fns, dist_path)
        cv2.imwrite(new_fn, box_blur)
        c += 8
        progress_bar.update(8)
        if c >= nb_of_image:
            break
    progress_bar.close()

if __name__ == '__main__':
    main()
