# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np



def smooth_l1_loss(predict, reference, positive_kind):

    delta_t = tf.cast(tf.abs(predict - reference), tf.float32)
    losses = tf.where(tf.less(delta_t, 1), 0.5 * tf.square(delta_t), delta_t - 0.5)
    return tf.reduce_mean(positive_kind * tf.reduce_sum(losses, axis=1), axis=0)

def smooth_l1_loss_(predict, reference, inside_weight):

    delta_t = tf.cast(inside_weight * tf.abs(predict - reference), tf.float32)
    losses = tf.where(tf.less(delta_t, 1), 0.5 * tf.square(delta_t), delta_t - 0.5)
    return tf.reduce_mean(tf.reduce_sum(losses, axis=1), axis=0)

def modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights):
    """
        ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
        SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                      |x| - 0.5 / sigma^2,    otherwise
    """
    sigma2 = sigma * sigma

    inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

    smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
    smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                              tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

    return tf.reduce_mean(tf.reduce_sum(smooth_l1_result, axis=1), axis=0)


if __name__=='__main__':
    predict = tf.constant([])
    #predict = tf.constant([1,2,3,4,5,6,7,8])
    predict = tf.reshape(predict, [-1, 4])
    reference = tf.constant([])
    #reference = tf.constant([8,7,6,5,4,3,2,1])
    reference = tf.reshape(reference, [-1, 4])
    a = tf.shape(predict)[0]

    sess = tf.Session()
    print(sess.run(a))
    print(sess.run(smooth_l1_loss(predict, reference)))



