# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np
from config import config as cfg


# aabb to xywh
def aabb_to_xywh(coordinate_aabb):

    xmin, ymin, xmax, ymax = tf.unstack(coordinate_aabb, axis=1)
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    coordinate_xywh = tf.stack([x_center, y_center, w, h], axis=1)

    return coordinate_xywh


def boxes_to_t(anchors, gtboxes):

    anchors_xywh = aabb_to_xywh(anchors)
    anchors_x, anchors_y, anchors_w, anchors_h = tf.unstack(anchors_xywh, axis=1)

    gtboxes_xywh = aabb_to_xywh(gtboxes)
    gtboxes_x, gtboxes_y, gtboxes_w, gtboxes_h = tf.unstack(gtboxes_xywh, axis=1)

    gtboxes_w += 1e-8
    gtboxes_h += 1e-8
    anchors_w += 1e-8
    anchors_h += 1e-8

    tx = (gtboxes_x - anchors_x) / anchors_w * cfg.T_SCALE[0]
    ty = (gtboxes_y - anchors_y) / anchors_h * cfg.T_SCALE[1]

    tw = tf.log(gtboxes_w / anchors_w) * cfg.T_SCALE[2]
    th = tf.log(gtboxes_h / anchors_h) * cfg.T_SCALE[3]

    return tf.stack([tx, ty, tw, th], axis=1)


def xywh_to_aabb(coordinate_xywh):

    x, y, w, h = tf.unstack(coordinate_xywh, axis=1)
    ymin = y - h / 2.
    ymax = y + h / 2.
    xmin = x - w / 2.
    xmax = x + w / 2.
    coordinate_aabb = tf.stack([xmin, ymin, xmax, ymax], axis=1)

    return coordinate_aabb


def t_to_boxes(anchors, t):

    anchors_xywh = aabb_to_xywh(anchors)
    anchors_x, anchors_y, anchors_w, anchors_h = tf.unstack(anchors_xywh, axis=1)
    tx, ty, tw, th = tf.unstack(t, axis=1)

    predict_x = anchors_w * tx / cfg.T_SCALE[0] + anchors_x
    predict_y = anchors_h * ty / cfg.T_SCALE[1] + anchors_y
    predict_w = anchors_w * tf.exp(tw / cfg.T_SCALE[2])
    predict_h = anchors_h * tf.exp(th / cfg.T_SCALE[3])

    predict_xywh = tf.stack([predict_x, predict_y, predict_w, predict_h], axis=1)
    predict_aabb = xywh_to_aabb(predict_xywh)

    return predict_aabb


