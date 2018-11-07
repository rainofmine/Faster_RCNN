# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

import numpy as np
from config import config as cfg


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 9]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           lambda: (tf.constant(target_shortside_len), target_shortside_len * w//h),
                           lambda: (target_shortside_len * h//w,  tf.constant(target_shortside_len)))
    '''
    if tf.less(h, w) is not None:
        new_h = target_shortside_len
        new_w = target_shortside_len * w // h
    else:
        new_h = target_shortside_len *h // w
        new_w = target_shortside_len
    '''

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    if cfg.LABEL_INCLINED:

        x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)

        x1, x2, x3, x4 = x1 * new_w//w, x2 * new_w//w, x3 * new_w//w, x4 * new_w//w
        y1, y2, y3, y4 = y1 * new_h//h, y2 * new_h//h, y3 * new_h//h, y4 * new_h//h

        img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
        return img_tensor, tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, label], axis=0))

    else:
        x1, y1, x2, y2, label = tf.unstack(gtboxes_and_label, axis=1)

        x1, x2 = x1 * new_w // w, x2 * new_w // w
        y1, y2 = y1 * new_h // h, y2 * new_h // h

        img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
        return img_tensor, tf.transpose(tf.stack([x1, y1, x2, y2, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)

    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               lambda: (tf.constant(target_shortside_len), target_shortside_len * w // h),
                               lambda: (target_shortside_len * h // w, tf.constant(target_shortside_len)))
        '''
        new_h, new_w = tf.cond(tf.less(h, w),
                               lambda: (target_shortside_len, target_shortside_len*w//h),
                               lambda: (target_shortside_len*h//w, target_shortside_len))
        '''
        '''
        if tf.less(h, w) is not None:
            new_h = target_shortside_len
            new_w = target_shortside_len * w // h
        else:
            new_h = target_shortside_len * h // w
            new_w = target_shortside_len
        '''

        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]


def random_flip_left_right(img_tensor, gtboxes_and_label):

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    coin = np.random.rand()
    if coin > 0.5:
        img_tensor = tf.image.flip_left_right(img_tensor)

        if cfg.LABEL_INCLINED:
            x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)
            new_x1 = w - x1
            new_x2 = w - x2
            new_x3 = w - x3
            new_x4 = w - x4
            return img_tensor, tf.transpose(tf.stack([new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, label], axis=0))
        else:
            x1, y1, x2, y2, label = tf.unstack(gtboxes_and_label, axis=1)
            new_x1 = w - x2
            new_x2 = w - x1
            return img_tensor, tf.transpose(tf.stack([new_x1, y1, new_x2, y2, label], axis=0))
    else:
        return img_tensor,  gtboxes_and_label


