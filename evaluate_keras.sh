#!/usr/bin bash

python extract_keras_features.py --images oxford/data/* --out oxford/keras_pool5
python extract_keras_queries.py --dataset oxford --out oxford/keras_pool5_queries
python extract_keras_features.py --images paris/data/* --out paris/keras_pool5
python extract_keras_queries.py --dataset paris --out paris/keras_pool5_queries

python evaluate.py --index_features oxford/keras_pool5 --whiten_features paris/keras_pool5 --queries oxford/keras_pool5_queries --wt ucrow
