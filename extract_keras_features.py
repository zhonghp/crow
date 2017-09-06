# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import time
import scipy
import numpy as np
from PIL import Image

import keras
import keras.backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input


###################################
# Feature Extraction
###################################

def load_img(path):
    """
    Load the image at the provided path and normalize to RGB.

    :param str path:
        path to image file
    :returns Image:
        Image object
    """
    try:
        return image.load_img(path)
    except:
        return None



def format_img_for_vgg(img):
    """
    Given an Image, convert to ndarray and preprocess for VGG.

    :param Image img:
        Image object
    :returns ndarray:
        3d tensor formatted for VGG
    """
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def extract_raw_features(model, d):
    """
    Extract raw features for a single image.
    """
    y = model.predict(d)[0]
    if K.image_data_format() == 'channels_last':
      return y.transpose((2, 0, 1))
    else:
      return y


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--images', dest='images', type=str, nargs='+', required=True, help='glob pattern to image data')
    parser.add_argument('--out', dest='out', type=str, default='', help='path to save output')
    parser.add_argument('--gpu', dest='gpu', type=str, default='1', help='gpu id to use')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = VGG16(weights='imagenet', include_top=False)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    beg = int(time.time())
    for path in args.images:
        img = load_img(path)

        # Skip if the image failed to load
        if img is None:
            print path
            continue

        d = format_img_for_vgg(img)
        X = extract_raw_features(model, d)

        filename = os.path.splitext(os.path.basename(path))[0]
        np.save(os.path.join(args.out, filename), X)
    end = int(time.time())
    print 'used_time:', (end-beg), 's'
