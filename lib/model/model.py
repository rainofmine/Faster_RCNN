# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from config import config as cfg


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_restorer():

    checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfg.SAVE_MODEL_PATH, cfg.VERSION))

    if checkpoint_path != None:
        restorer = tf.train.Saver()
        print("find last model, model restore from :", checkpoint_path)
    else:
        rm_list = os.listdir(os.path.join(cfg.SUMMARY_PATH, cfg.VERSION))
        for i in rm_list:
            os.remove(os.path.join(cfg.SUMMARY_PATH, cfg.VERSION, i))

        checkpoint_path = cfg.PRETRAINED_MODEL_PATH
        print("feature model restore :", checkpoint_path)

        model_variables = slim.get_model_variables()

        restore_variables = [var for var in model_variables
                             if (var.name.startswith(cfg.NET_NAME)
                                 and not var.name.startswith('{}/logits'.format(cfg.NET_NAME))
                                 and not var.name.startswith('{}/fc'.format(cfg.NET_NAME))
                                 #and var.name.find('BatchNorm')==-1
                                )
                             ]
        for var in restore_variables:
            print(var.name)
        restorer = tf.train.Saver(restore_variables)
    return restorer, checkpoint_path