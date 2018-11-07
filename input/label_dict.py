# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from config import config as cfg

if cfg.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'ship': 1
    }
elif cfg.DATASET_NAME == 'aeroplane':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1
    }
elif cfg.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
elif cfg.DATASET_NAME == 'hooks':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        't-insulator': 1,
        'b-insulator': 2,
        'c-support': 3,
        'ff-tube': 4,
        'ft-support': 5,
        'ff-device': 6,
        'connector': 7
    }
else:
    assert 'please set label dict!'


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()
