import importlib
import inspect
import os
import sys

import bs4
import numpy as np
import pandas as pandas
import tensorflow as tf
import tqdm

import utils.dataset.cifar10 as cifar10
import utils.dataset.imagenet as imagenet
import utils.dataset.voc as voc
import utils.tfimage as tfimage
import utils.tfsys as tfsys

datasets_map = {'cifar10': cifar10,
                'imagenet': imagenet,
                'voc': voc,
                }


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None):
    if name not in datasets_map:
        raise ValueError('%s dataset is unknown' % name)
    return datasets_map[name].get_split(split_name, dataset_dir, file_pattern, reader)


def download_convert(config, args):
    if args.dataset_name not in datasets_map:
        raise ValueError('%s dataset is unkown' % args.name)
    datasets_map[args.dataset_name].convert_to_tfrecord(config)