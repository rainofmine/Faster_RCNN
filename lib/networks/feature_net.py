# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import vgg
from config import config as cfg


def get_featuremap(net_name, input, num_classes=None):

    '''
    #tensorlayer
    input = tl.layers.InputLayer(input)
    if net_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.FEATURE_WEIGHT_DECAY)):
            featuremap = tl.layers.SlimNetsLayer(prev_layer=input,
                                                 slim_layer=resnet_v1.resnet_v1_50,
                                                 slim_args={
                                                     'num_classes': num_classes,
                                                     'is_training': True,
                                                     'global_pool': False
                                                 },
                                                 name='resnet_v1_50'
                                                 )
            sv = tf.train.Supervisor()
            with sv.managed_session() as sess:
                a = sess.run(featuremap.all_layers)
                print(a)
            feature_w_loss = tf.reduce_sum(slim.losses.get_regularization_losses())
            return featuremap.outputs, feature_w_loss, featuremap.all_params
    if net_name == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            featuremap = tl.layers.SlimNetsLayer(prev_layer=input,
                                                 slim_layer=resnet_v1.resnet_v1_101,
                                                 slim_args={
                                                     'num_classes': num_classes,
                                                     'is_training': True,
                                                     'global_pool': False
                                                 },
                                                 name='resnet_v1_101'
                                                 )
            feature_w_loss = tf.reduce_sum(slim.losses.get_regularization_losses())
            return featuremap.outputs, feature_w_loss, featuremap.all_params
    if net_name == 'resnet_v1_152':
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            featuremap = tl.layers.SlimNetsLayer(prev_layer=input,
                                                 slim_layer=resnet_v1.resnet_v1_152,
                                                 slim_args={
                                                     'num_classes': num_classes,
                                                     'is_training': True,
                                                     'global_pool': False
                                                 },
                                                 name='resnet_v1_152'
                                                 )
            feature_w_loss = tf.reduce_sum(slim.losses.get_regularization_losses())
            return featuremap.outputs, feature_w_loss, featuremap.all_params
    if net_name == 'vgg16':
        with slim.arg_scope(vgg.vgg_arg_scope()):
            featuremap = tl.layers.SlimNetsLayer(prev_layer=input,
                                                 slim_layer=vgg.vgg_16,
                                                 slim_args={
                                                     'num_classes': num_classes,
                                                     'is_training': True,
                                                     'spatial_squeeze': False
                                                 },
                                                 name='vgg_16'
                                                 )
            feature_w_loss = tf.reduce_sum(slim.losses.get_regularization_losses())
            return featuremap.outputs, feature_w_loss, featuremap.all_params
    '''


    #slim
    if net_name == 'resnet_v1_50':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.FEATURE_WEIGHT_DECAY)):
            featuremap, layer_dic = resnet_v1.resnet_v1_50(inputs=input,
                                                           num_classes=num_classes,
                                                           is_training=False,
                                                           global_pool=False
                                                           )
        if cfg.USE_FPN:
            feature_maps_dict = {
                'C2': layer_dic['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # [56, 56]
                'C3': layer_dic['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # [28, 28]
                'C4': layer_dic['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # [14, 14]
                'C5': layer_dic['resnet_v1_50/block4']  # [7, 7]
            }
            return feature_maps_dict
        return layer_dic['resnet_v1_50/block3/unit_5/bottleneck_v1']
        #return featuremap

    if net_name == 'resnet_v1_101':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.FEATURE_WEIGHT_DECAY)):
            featuremap, layer_dic = resnet_v1.resnet_v1_101(inputs=input,
                                                            num_classes=num_classes,
                                                            is_training=True,
                                                            global_pool=False
                                                            )
        if cfg.USE_FPN:
            feature_maps_dict = {
                'C2': layer_dic['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # [56, 56]
                'C3': layer_dic['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # [28, 28]
                'C4': layer_dic['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # [14, 14]
                'C5': layer_dic['resnet_v1_101/block4']
            }
            return feature_maps_dict
        return featuremap

    if net_name == 'vgg_16':
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=cfg.FEATURE_WEIGHT_DECAY)):
            featuremap, layer_dic = vgg.vgg_16(inputs=input,
                                               num_classes=7,
                                               is_training=False,
                                               spatial_squeeze=False,
                                               )

        return layer_dic['vgg_16/conv5/conv5_3']

