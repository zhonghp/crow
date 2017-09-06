# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import os
import glob
import time
import numpy as np
from PIL import Image

import keras
from keras.models import Model
from keras.applications.vgg16 import VGG16
from extract_keras_features import format_img_for_vgg, extract_raw_features


def query_images(groundtruth_dir, image_dir, dataset, cropped=True):
    """
    Extract features from the Oxford or Paris dataset.

    :param str groundtruth_dir:
        the directory of the groundtruth files (which includes the query files)
    :param str image_dir:
        the directory of dataset images
    :param str dataset:
        the name of the dataset, either 'oxford' or 'paris'
    :param bool cropped:
        flag to optionally disable cropping

    :yields Image img:
        the Image object
    :yields str query_name:
        the name of the query
    """
    for f in glob.iglob(os.path.join(groundtruth_dir, '*_query.txt')):
        query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        img_name, x, y, w, h = open(f).read().strip().split(' ')

        if dataset == 'oxford':
            img_name = img_name.replace('oxc1_', '')
        img = Image.open(os.path.join(image_dir, '%s.jpg' % img_name))

        if cropped:
            x, y, w, h = map(float, (x, y, w, h))
            box = map(lambda d: int(round(d)), (x, y, x + w, y + h))
            img = img.crop(box)

        yield img, query_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True, help='dataset to extract queries for')
    parser.add_argument('--images', dest='images', type=str, default='data/', help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='groundtruth/', help='directory containing groundtruth files')
    parser.add_argument('--out', dest='out', type=str, default='pool5_queries/', help='path to save output')
    parser.add_argument('--gpu', dest='gpu', type=str, default='1', help='gpu id to use')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = VGG16(weights='imagenet', include_top=False)

    images_dir = os.path.join(args.dataset, args.images)
    groundtruth_dir = os.path.join(args.dataset, args.groundtruth)
    out_dir = os.path.join(args.dataset, args.out)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    beg = int(time.time())
    for img, name in query_images(groundtruth_dir, images_dir, args.dataset):
        d = format_img_for_vgg(img)
        X = extract_raw_features(model, d)

        np.save(os.path.join(out_dir, '%s' % name), X)
    end = int(time.time())
    print 'used_time:', (end-beg), 's'
