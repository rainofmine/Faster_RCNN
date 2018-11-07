# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np

from config import config as cfg
from lib.boxes import anchor, box_regression
from lib.iou import iou
from lib.loss.location_loss import smooth_l1_loss
from lib.boxes.box_cv import draw_box_with_color


class RPN:
    def __init__(self, img_batch, img_shape, gtboxes_and_label, featuremap, is_training):
        self.img_batch = img_batch
        self.net_name = cfg.NET_NAME
        self.img_shape = img_shape
        self.gtboxes_and_label = gtboxes_and_label
        self.featuremap = featuremap
        self.is_training = is_training

        self.stride = cfg.STRIDE
        self.level = cfg.LEVEL
        self.base_anchor_size_list = cfg.BASE_ANCHOR_SIZE_LIST

        self.anchor_scales = tf.constant(cfg.ANCHOR_SCALES)
        self.anchor_ratios = tf.constant(cfg.ANCHOR_RATIOS)
        self.anchor_k = len(cfg.ANCHOR_SCALES) * len(cfg.ANCHOR_RATIOS)

        self.remove_outside_anchors = cfg.REMOVE_OUTSIDES_ANCHORS

        self.positive_threshold = cfg.RPN_POSITIVE_THRESHOLD
        self.negative_threshold = cfg.RPN_NEGATIVE_THRESHOLD

        self.minibatch_size = cfg.RPN_MINIBATCH_SIZE
        self.minibatch_positive_ratio = cfg.RPN_MINIBATCH_POSITIVE_RATIO

        self.topk_num = cfg.TOPK_NUM
        self.rpn_num = cfg.RPN_NUM
        self.nms_threshold = cfg.NMS_THRESHOLD

        self.feature_pyramid = None
        if cfg.USE_FPN:
            self.feature_pyramid = self.build_fpn()
            self.anchor_k = len(cfg.ANCHOR_RATIOS)

        self.anchors, self.rpn_scores, self.rpn_coordinates_t = self.rpn_inference()



    def build_fpn(self):

        feature_pyramid = {}
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfg.FEATURE_WEIGHT_DECAY)):
            feature_pyramid['P5'] = slim.conv2d(self.featuremap['C5'],
                                                num_outputs=256,
                                                kernel_size=[1, 1],
                                                stride=1,
                                                scope='build_P5')
            feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                    kernel_size=[2, 2], stride=2, scope='build_P6')

            for layer in range(4, 1, -1):
                p, c = feature_pyramid['P' + str(layer + 1)], self.featuremap['C' + str(layer)]
                up_sample_shape = tf.shape(c)
                up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                             name='build_P%d/up_sample_nearest_neighbor' % layer)

                c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                scope='build_P%d/reduce_dimension' % layer)
                p = up_sample + c
                p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                padding='SAME', scope='build_P%d/avoid_aliasing' % layer)
                feature_pyramid['P' + str(layer)] = p

        return feature_pyramid

    def make_anchors(self):

        if cfg.USE_FPN:
            anchor_list = []
            for level, base_anchor_size, stride in zip(self.level, self.base_anchor_size_list, self.stride):
                featuremap_h, featuremap_w = tf.shape(self.feature_pyramid[level])[1], \
                                                      tf.shape(self.feature_pyramid[level])[2]
                anchors = anchor.make_anchors(featuremap_h, featuremap_w, [base_anchor_size], self.anchor_ratios, stride)
                anchors = tf.reshape(anchors, [-1, 4])
                anchor_list.append(anchors)
            anchors = tf.concat(anchor_list, axis=0)

        else:
            featuremap_h = tf.shape(self.featuremap)[1]
            featuremap_w = tf.shape(self.featuremap)[2]
            anchors = anchor.make_anchors(featuremap_h, featuremap_w, self.anchor_scales, self.anchor_ratios, self.stride)

        return anchors


    def rpn_inference(self):
        '''
        featuremap = tl.layers.InputLayer(self.featuremap)

        rpn_3x3 = tl.layers.Conv2d(featuremap,
                                   n_filter=256,
                                   filter_size=(3,3),
                                   strides=(1,1),
                                   act=tf.nn.relu,
                                   name='rpn_3x3'
                                   )
        rpn_1x1_scores = tl.layers.Conv2d(rpn_3x3,
                                          n_filter=2 * self.anchor_k,
                                          filter_size=(1,1),
                                          strides=(1,1),
                                          act=tf.nn.relu,
                                          name='rpn_1x1_scores'
                                          )
        rpn_1x1_coordinates_t = tl.layers.Conv2d(rpn_3x3,
                                                 n_filter=4 * self.anchor_k,
                                                 filter_size=(1, 1),
                                                 strides=(1, 1),
                                                 act=tf.nn.relu,
                                                 name='rpn_1x1_coordinates_t'
                                                 )
        rpn_scores = tf.reshape(rpn_1x1_scores.outputs, [-1, 2])
        rpn_coordinates_t = tf.reshape(rpn_1x1_coordinates_t.outputs, [-1, 4])
        '''

        if cfg.USE_FPN:
            rpn_boxes_list = []
            rpn_scores_list = []
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfg.FEATURE_WEIGHT_DECAY)):
                for level in self.level:
                    scope_list = ['conv2d_3x3_' + level, 'rpn_classifier_' + level, 'rpn_regressor_' + level]
                    rpn_3x3 = slim.conv2d(inputs=self.feature_pyramid[level],
                                                 num_outputs=256,
                                                 kernel_size=[3, 3],
                                                 stride=1,
                                                 scope=scope_list[0],
                                                 reuse=None)
                    rpn_1x1_scores = slim.conv2d(rpn_3x3,
                                                 num_outputs=2 * self.anchor_k,
                                                 kernel_size=[1, 1],
                                                 stride=1,
                                                 scope=scope_list[1],
                                                 activation_fn=None,
                                                 reuse=None)
                    rpn_1x1_coordinates_t = slim.conv2d(rpn_1x1_scores,
                                                   num_outputs=4 * self.anchor_k,
                                                   kernel_size=[1, 1],
                                                   stride=1,
                                                   scope=scope_list[2],
                                                   activation_fn=None,
                                                   reuse=None)
                    rpn_scores = tf.reshape(rpn_1x1_scores, [-1, 2])
                    rpn_boxes = tf.reshape(rpn_1x1_coordinates_t, [-1, 4])

                    rpn_scores_list.append(rpn_scores)
                    rpn_boxes_list.append(rpn_boxes)

                rpn_coordinates_t = tf.concat(rpn_boxes_list, axis=0)
                rpn_scores = tf.concat(rpn_scores_list, axis=0)

        else:
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(cfg.FEATURE_WEIGHT_DECAY)):
                rpn_3x3 = slim.conv2d(inputs=self.featuremap,
                                      num_outputs=256,
                                      kernel_size=[3, 3],
                                      stride=1,
                                      scope='rpn_3x3',
                                      reuse=None)
                rpn_1x1_scores = slim.conv2d(inputs=rpn_3x3,
                                             num_outputs=2 * self.anchor_k,
                                             kernel_size=[1, 1],
                                             stride=1,
                                             scope='rpn_1x1_scores',
                                             activation_fn=None,
                                             reuse=None)
                rpn_1x1_coordinates_t = slim.conv2d(inputs=rpn_3x3,
                                                    num_outputs=4 * self.anchor_k,
                                                    kernel_size=[1, 1],
                                                    stride=1,
                                                    scope='rpn_1x1_coordinates_t',
                                                    activation_fn=None,
                                                    reuse=None
                                                    )
            rpn_scores = tf.reshape(rpn_1x1_scores, [-1, 2])
            rpn_coordinates_t = tf.reshape(rpn_1x1_coordinates_t, [-1, 4])

        anchors = self.make_anchors()

        if self.is_training:
            if self.remove_outside_anchors:
                img_h = tf.cast(self.img_shape[1], dtype=tf.float32)
                img_w = tf.cast(self.img_shape[2], dtype=tf.float32)
                anchors_valid_index = anchor.find_inside_anchors(anchors, img_w, img_h)
                anchors_valid = tf.gather(anchors, anchors_valid_index)
                rpn_scores_valid = tf.gather(rpn_scores, anchors_valid_index)
                rpn_coordinates_t_valid = tf.gather(rpn_coordinates_t, anchors_valid_index)
                return anchors_valid, rpn_scores_valid, rpn_coordinates_t_valid

        return anchors, rpn_scores, rpn_coordinates_t


    def define_samples(self):

        #positive 1 or 2, negative 0, ignored -1
        sample_kind = tf.ones(shape=[tf.shape(self.anchors)[0] , ] ,dtype=tf.int32) * (-1)

        gtboxes = tf.cast(self.gtboxes_and_label[:, :-1], dtype=tf.float32)
        iou_each_anchor = iou.iou_calculate(self.anchors, gtboxes)

        iou_each_anchor_max = tf.reduce_max(iou_each_anchor, axis=1)

        negative_index = tf.less(iou_each_anchor_max, self.negative_threshold)
        #negative_index = tf.logical_and(negative_index, tf.greater_equal(iou_each_anchor_max, 0.1))

        sample_kind = sample_kind + tf.cast(negative_index, tf.int32)

        positive_index_1 = tf.greater_equal(iou_each_anchor_max, self.positive_threshold)

        iou_gt_max_index = tf.reduce_max(iou_each_anchor, axis=0)
        positive_index_2 = tf.cast(tf.reduce_sum(tf.cast(tf.equal(iou_each_anchor, iou_gt_max_index), tf.int32), axis=1), tf.bool)

        positive_index = tf.logical_or(positive_index_1, positive_index_2)

        sample_kind = sample_kind + tf.cast(positive_index, tf.int32) * 2

        # positive 1, negative 0, ignored -1
        positive_index = tf.cast(tf.greater_equal(sample_kind, 1), tf.int32)
        ignored_index = (-1) * tf.cast(tf.equal(sample_kind, -1), tf.int32)

        positive_kind = tf.cast(positive_index, tf.float32)
        sample_kind = positive_index + ignored_index

        #the gtboxes reference to each anchor
        gtboxes_index = tf.argmax(iou_each_anchor, axis=1)
        gtboxes_for_each_anchor = tf.gather(gtboxes, gtboxes_index)

        return positive_kind, sample_kind, gtboxes_for_each_anchor


    def get_minibatch(self):

        positive_kind, sample_kind, gtboxes_for_each_anchor = self.define_samples()

        positive_index = tf.reshape(tf.where(tf.equal(sample_kind, 1)), [-1])
        positive_index = tf.random_shuffle(positive_index)
        positive_num = tf.minimum(tf.shape(positive_index)[0],
                                  tf.cast((self.minibatch_size * self.minibatch_positive_ratio), tf.int32))
        positive_minibatch_index = tf.slice(positive_index, begin=[0], size=[positive_num])

        negative_index = tf.reshape(tf.where(tf.equal(sample_kind, 0)), [-1])
        negative_index = tf.random_shuffle(negative_index)
        negative_num = tf.minimum(tf.shape(negative_index)[0],
                                  tf.cast(self.minibatch_size, tf.int32) - positive_num)
        negative_minibatch_index = tf.slice(negative_index, begin=[0], size=[negative_num])

        minibatch_index = tf.concat([positive_minibatch_index, negative_minibatch_index], axis=0)
        minibatch_index = tf.random_shuffle(minibatch_index)

        minibatch_labels = tf.gather(sample_kind, minibatch_index)
        minibatch_gtboxes = tf.gather(gtboxes_for_each_anchor, minibatch_index)

        return positive_kind, minibatch_index, minibatch_labels, minibatch_gtboxes


    def rpn_loss(self):

        positive_kind, minibatch_index, minibatch_labels, minibatch_gtboxes = self.get_minibatch()

        positive_kind = tf.gather(positive_kind, minibatch_index)
        minibatch_anchors = tf.gather(self.anchors, minibatch_index)
        minibatch_scores = tf.gather(self.rpn_scores, minibatch_index)
        minibatch_coornadite_t = tf.gather(self.rpn_coordinates_t, minibatch_index)

        #minibatch_anchors_img = draw_box_with_color(self.img_batch, minibatch_anchors, 0)
        positive_anchors_img = draw_box_with_color(self.img_batch, minibatch_anchors * tf.expand_dims(positive_kind, 1), 0)
        #tf.summary.image('minibatch_anchors', minibatch_anchors_img)
        tf.summary.image('positive_anchors', positive_anchors_img)

        regression_vector = box_regression.boxes_to_t(minibatch_anchors, minibatch_gtboxes)
        #gtbox_img = box_regression.t_to_boxes(minibatch_anchors, regression_vector)
        #gtbox_img = draw_box_with_color(self.img_batch, gtbox_img, 0)
        #tf.summary.image('gtbox_train', gtbox_img)

        #classification_loss = tl.cost.cross_entropy(minibatch_scores, minibatch_labels, name='rpn_class_loss')
        classification_loss = slim.losses.sparse_softmax_cross_entropy(minibatch_scores, minibatch_labels)
        location_loss = cfg.RPN_LOSS_LAMBDA * smooth_l1_loss(minibatch_coornadite_t, regression_vector, positive_kind)
        #slim.losses.add_loss(location_loss)

        return classification_loss, location_loss


    def get_rpn(self):

        predict_boxes = box_regression.t_to_boxes(self.anchors, self.rpn_coordinates_t)

        rpn_scores = tf.nn.softmax(self.rpn_scores)
        foreground_score = rpn_scores[:, 1]

        top20_rpn_scores, top20_rpn_index = tf.nn.top_k(foreground_score, 20)
        top20_rpn_img = draw_box_with_color(self.img_batch, tf.gather(predict_boxes, top20_rpn_index), 20)
        tf.summary.image('top20_rpn', top20_rpn_img)

        select_rpn_scores, select_index = tf.nn.top_k(foreground_score, self.topk_num)
        select_boxes = tf.gather(predict_boxes, select_index)

        final_index = tf.image.non_max_suppression(select_boxes, select_rpn_scores, self.rpn_num, self.nms_threshold)
        final_rpn_scores = tf.gather(select_rpn_scores, final_index)
        final_boxes = tf.gather(select_boxes, final_index)

        '''
        select_index = tf.image.non_max_suppression(predict_boxes, foreground_score, self.rpn_num, self.nms_threshold)
        select_rpn_scores = tf.gather(foreground_score, select_index)
        select_boxes = tf.gather(predict_boxes, select_index)

        final_rpn_scores, final_index = tf.nn.top_k(select_rpn_scores, 2000)
        final_boxes = tf.gather(select_boxes, final_index)
        '''
        top300_rpn_scores, top300_rpn_index = tf.nn.top_k(final_rpn_scores, 300)
        top300_rpn_img = draw_box_with_color(self.img_batch, tf.gather(final_boxes, top300_rpn_index), 300)
        tf.summary.image('top300_rpn', top300_rpn_img)

        return final_rpn_scores, final_boxes









