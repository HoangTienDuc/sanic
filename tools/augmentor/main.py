import argparse
import os
import glob
import cv2
import random

import scipy
from tqdm import tqdm
import numpy as np
from Augmentor import Operations
import albumentations as A


def add_noise_sp(im):
    row, col, ch = im.shape
    s_vs_p = 0.5
    amount = 0.006
    out = np.copy(im)

    # Salt mode
    num_salt = np.ceil(amount * im.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in im.shape]
    out[coords] = 1
    # Pepper mode
    num_pepper = np.ceil(amount * im.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in im.shape]
    out[coords] = 0

    return out


def blur(im):
    return cv2.medianBlur(im, 5)


def bilateral(im):
    return cv2.bilateralFilter(im, 15, 18.0, 32.0)


def get_aug(aug, min_area=0., min_visibility=0.):
    return A.Compose(aug, bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})

def cbs(im):
    aug = get_aug([
        A.RGBShift(p=0.5),
        A.RandomBrightness(p=0.5),
        A.CLAHE(p=0.5),
    ])
    annotations = {'image': im, 'bboxes': [[366.7, 80.84, 132.8, 181.84], [5.66, 138.95, 147.09, 164.88]], 'category_id': [18, 17]}
    augmented = aug(**annotations)
    return augmented['image'].copy()


def solveLinearEquation(IN, wx, wy, lamda):
    [r, c] = IN.shape
    k = r * c
    dx =  -lamda * wx.flatten('F')
    dy =  -lamda * wy.flatten('F')
    tempx = np.roll(wx, 1, axis=1)
    tempy = np.roll(wy, 1, axis=0)
    dxa = -lamda *tempx.flatten('F')
    dya = -lamda *tempy.flatten('F')
    tmp = wx[:,-1]
    tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
    tmp = wy[-1,:]
    tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
    dxd1 = -lamda * tempx.flatten('F')
    dyd1 = -lamda * tempy.flatten('F')

    wx[:,-1] = 0
    wy[-1,:] = 0
    dxd2 = -lamda * wx.flatten('F')
    dyd2 = -lamda * wy.flatten('F')

    Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
    Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
    D = 1 - ( dx + dy + dxa + dya)
    A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T

    tin = IN[:,:]
    tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
    OUT = np.reshape(tout, (r, c), order='F')

    return OUT

def computeTextureWeights(fin, sigma, sharpness):
    dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
    dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

    gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,sigma)), mode='same')
    gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((sigma,1)), mode='same')

    W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h)+sharpness)
    W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v)+sharpness)

    return  W_h, W_v

def tsmooth(img, lamda=0.01, sigma=3.0, sharpness=0.001):
    I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    x = np.copy(I)
    wx, wy = computeTextureWeights(x, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lamda)
    return S

def rgb2gm(I):
    if (I.shape[2] == 3):
        I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        I = (I[:,:,0]*I[:,:,1]*I[:,:,2])**(1/3)

    return I

def applyK(I, k, a=-0.3293, b=1.1258):
    f = lambda x: np.exp((1-x**a)*b)
    beta = f(k)
    gamma = k**a
    J = (I**gamma)*beta
    return J

def entropy(X):
    tmp = X * 255
    tmp[tmp > 255] = 255
    tmp[tmp<0] = 0
    tmp = tmp.astype(np.uint8)
    _, counts = np.unique(tmp, return_counts=True)
    pk = np.asarray(counts)
    pk = 1.0*pk / np.sum(pk, axis=0)
    S = -np.sum(pk * np.log2(pk), axis=0)
    return S

def maxEntropyEnhance(I, isBad, a=-0.3293, b=1.1258):
    # Esatimate k
    tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
    tmp[tmp<0] = 0
    tmp = tmp.real
    Y = rgb2gm(tmp)

    isBad = isBad * 1
    isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')
    isBad[isBad<0.5] = 0
    isBad[isBad>=0.5] = 1
    Y = Y[isBad==1]

    if Y.size == 0:
       J = I
       return J

    f = lambda k: -entropy(applyK(Y, k))
    opt_k = scipy.optimize.fminbound(f, 1, 7)

    # Apply k
    J = applyK(I, opt_k, a, b) - 0.01
    return J

def Ying_2017_CAIP(im):
    lamda = 0.5
    sigma = 5
    mu = 0.5
    a = -0.3293
    b = 1.1258

    I = cv2.normalize(im.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Weight matrix estimation
    t_b = np.max(I, axis=2)
    try:
        t_our = cv2.resize(tsmooth(scipy.misc.imresize(t_b, 0.5, interp='bicubic', mode='F'), lamda, sigma), (t_b.shape[1], t_b.shape[0]), interpolation=cv2.INTER_AREA)

        # Apply camera model with k(exposure ratio)
        isBad = t_our < 0.5
        J = maxEntropyEnhance(I, isBad)

        # W: Weight Matrix
        t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
        for i in range(I.shape[2]):
            t[:,:,i] = t_our
        W = t**mu

        I2 = I*W
        J2 = J*(1-W)

        result = I2 + J2
        result = result * 255
        result[result > 255] = 255
        result[result<0] = 0

        return result.astype(np.uint8)
    except ValueError:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_file', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    im_names, labels = [], []
    for line in open(args.label_file):
        try:
            im_name, label = line[:-1].split('\t')
        except:
            continue
        im_names.append(im_name)
        labels.append(label)
    labels = np.array(labels)

    im_paths = [os.path.join(args.input_dir, im_name) for im_name in im_names]
    im_l = [cv2.imread(im) for im in im_paths]
    im_l = np.array(list(filter(lambda x: x is not None, im_l)))
    aug_method = {
        'noisy': add_noise_sp,
        'blur': blur,
        'bilat': bilateral,
        'cbs': cbs,
        'ying': Ying_2017_CAIP
    }

    with open(os.path.join(args.output_dir, 'label.txt'), 'w') as f:
        for cond, method in tqdm(aug_method.items()):
            rand_id = random.sample(range(len(im_l)), 20000)
            rand_labels = labels[rand_id]
            im_aug = map(method, im_l[rand_id])

            for im, name, label in zip(im_aug, im_names, rand_labels):
                if im is not None:
                    im_dir = os.path.join(args.output_dir, 'img')
                    os.makedirs(im_dir, exist_ok=True)
                    dest_name = cond+'_'+name
                    cv2.imwrite(os.path.join(im_dir, dest_name), im)
                    f.write('{}\t{}\n'.format(dest_name, label))
