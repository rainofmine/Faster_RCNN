# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np



def make_anchors(featuremap_h, featuremap_w, anchor_scales, anchor_ratios, stride):

    anchor_scales = tf.reshape(anchor_scales, [1, -1])
    sqrt_ratios = tf.sqrt(anchor_ratios)
    sqrt_ratios = tf.reshape(sqrt_ratios, [-1, 1])
    hs = anchor_scales * sqrt_ratios
    ws = anchor_scales / sqrt_ratios

    hs = tf.squeeze(tf.reshape(hs, [1, -1]))
    ws = tf.squeeze(tf.reshape(ws, [1, -1]))

    xs = tf.range(tf.cast(featuremap_w, tf.float32), dtype=tf.float32) * stride
    ys = tf.range(tf.cast(featuremap_h, tf.float32), dtype=tf.float32) * stride

    xs, ys = tf.meshgrid(xs, ys)
    ws, xs = tf.meshgrid(ws, xs)
    hs, ys = tf.meshgrid(hs, ys)

    box_centers = tf.stack([xs, ys], axis=2)
    box_centers = tf.reshape(box_centers, [-1, 2])

    box_sizes = tf.stack([ws, hs], axis=2)
    box_sizes = tf.reshape(box_sizes, [-1, 2])

    anchors = tf.concat([box_centers - 0.5 * box_sizes, box_centers + 0.5 * box_sizes], axis=1)

    return anchors


def find_inside_anchors(anchors, img_w, img_h):

    xmin, ymin, xmax, ymax = tf.unstack(anchors, axis=1)
    ymin_inside = tf.greater_equal(ymin, 0)
    xmin_inside = tf.greater_equal(xmin, 0)
    ymax_inside = tf.less_equal(ymax, img_h)
    xmax_inside = tf.less_equal(xmax, img_w)

    anchors_valid_side = tf.transpose(tf.stack([xmin_inside, ymin_inside, xmax_inside, ymax_inside], axis=0))
    anchors_valid_side = tf.cast(anchors_valid_side, dtype=tf.int32)
    anchors_valid = tf.reduce_sum(anchors_valid_side, axis=1)
    anchors_valid_index = tf.where(tf.equal(anchors_valid, tf.shape(anchors)[1]))
    anchors_valid_index = tf.reshape(anchors_valid_index, [-1, ])

    return anchors_valid_index







if __name__ == '__main__':
    anchor_scales = tf.constant([128, 256, 512], dtype=tf.float32)
    anchor_ratios = tf.constant([1/2, 1, 2])
    anchors = make_anchors(100, 60, anchor_scales, anchor_ratios, 6)
    valid = find_inside_anchors(anchors, 1000, 600)
    index = tf.gather(anchors, valid)
    sess = tf.Session()
    print(sess.run(anchors))
    print(sess.run(valid))
    print(sess.run(index))