# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np


def boundary_process(boxes, img_shape):

    xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)
    img_h, img_w = img_shape[1], img_shape[2]

    xmin = tf.maximum(xmin, 0.0)
    xmin = tf.minimum(xmin, tf.cast(img_w, tf.float32))

    ymin = tf.maximum(ymin, 0.0)
    ymin = tf.minimum(ymin, tf.cast(img_h, tf.float32))  # avoid xmin > img_w, ymin > img_h

    xmax = tf.minimum(xmax, tf.cast(img_w, tf.float32))
    ymax = tf.minimum(ymax, tf.cast(img_h, tf.float32))

    return tf.stack([xmin, ymin, xmax, ymax], axis=1)