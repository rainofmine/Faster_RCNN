# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np

from config import config as cfg
from lib.iou import iou
from lib.boxes import box_regression, boundary_process
from lib.loss.location_loss import smooth_l1_loss, smooth_l1_loss_, modified_smooth_l1
from lib.boxes.box_cv import draw_box_with_color


def _find_match_t(t, labels):

    t_list = []
    for i in range(t.shape[0]):
        '''
        if labels[i] == 0:
            t_each_roi = np.array([0, 0, 0, 0]).astype(np.float32)
        else:
            t_each_roi = t[i][4 * (labels[i] - 1): (4 * (labels[i] - 1) + 4)]
        '''
        t_each_roi = t[i][(4 * labels[i]): (4 * labels[i] + 4)]
        t_list.append(t_each_roi)
    match_t = np.array(t_list)
    return match_t


def _get_train_t(t, labels, class_num):

    num_roi = labels.shape[0]
    train_t = np.zeros(shape=[num_roi, 4*(class_num+1)], dtype=np.float32)
    inside_weight = np.zeros(shape=train_t.shape, dtype=np.float32)
    for i in range(num_roi):
        if labels[i] > 0.:
            train_t[i][4 * (labels[i]): (4 * (labels[i]) + 4)] = t[i]
            inside_weight[i][4 * labels[i]: (4 * labels[i] + 4)] = [1., 1., 1., 1.]
    return train_t, inside_weight



class FAST_RCNN:
    def __init__(self, img_batch, img_shape, gtboxes_and_label, featuremap, feature_pyramid, rpn_scores, rpn_boxes, is_training):
        self.img_batch = img_batch
        self.img_shape = img_shape
        self.gtboxes_and_label = gtboxes_and_label
        self.featuremap = featuremap
        self.feature_pyramid = feature_pyramid
        self.rpn_scores = rpn_scores
        self.rpn_boxes = rpn_boxes
        self.is_training = is_training

        self.level = cfg.LEVEL
        self.min_level = int(self.level[0][1])
        self.max_level = min(int(self.level[-1][1]), 5)
        self.roi_size = cfg.ROI_SIZE
        self.class_num = cfg.CLASS_NUM
        self.positive_threshold = cfg.FAST_RCNN_POSITIVE_THRESHOLD
        self.minibatch_size = cfg.FAST_RCNN_MINIBATCH_SIZE
        self.minibatch_positive_ratio = cfg.FAST_RCNN_MINIBATCH_POSITIVE_RATIO
        self.boxes_num_each_class = cfg.BOX_NUM_EACH_CLASS
        self.nms_threshold_each_class = cfg.NMS_THRESHOLD_EACH_CLASS
        self.show_boxes_threshold = cfg.SHOW_BOXES_THRESHOLD

        self.roi, self.rpn_boxes = self.ROI_pooling()
        self.scores, self.coordinate_t = self.fast_rcnn_inference()


    def asign_level(self):

        xmin, ymin, xmax, ymax = tf.unstack(self.rpn_boxes, axis=1)

        w = tf.maximum(xmax - xmin, 0)
        h = tf.maximum(ymax - ymin, 0)

        k = tf.round(4. + tf.log(tf.sqrt(w*h + 1e-8)/224) / tf.log(2.))

        k = tf.maximum(k, tf.ones_like(k) * (np.float32(self.min_level)))
        k = tf.minimum(k, tf.ones_like(k) * (np.float32(self.max_level)))

        return tf.cast(k, tf.int32)


    def ROI_pooling(self):

        img_h = tf.cast(self.img_shape[1], tf.float32)
        img_w = tf.cast(self.img_shape[2], tf.float32)

        if cfg.USE_FPN:
            k = self.asign_level()
            level_rpn_boxes_list = []
            level_cropped_rois_list = []
            for i in range(self.min_level, self.max_level):
                level_index = tf.reshape(tf.where(tf.equal(k, i)), [-1])
                level_rpn_boxes = tf.gather(self.rpn_boxes, level_index)

                level_rpn_boxes = tf.cond(tf.equal(tf.shape(level_rpn_boxes)[0], 0),
                                          lambda: tf.constant([[0, 0, 0, 0]], tf.float32),
                                          lambda: level_rpn_boxes)
                level_rpn_boxes_list.append(level_rpn_boxes)

                xmin, ymin, xmax, ymax = tf.unstack(level_rpn_boxes, axis=1)
                unit_ymin = ymin / img_h
                unit_xmin = xmin / img_w
                unit_ymax = ymax / img_h
                unit_xmax = xmax / img_w

                level_cropped_rois = tf.image.crop_and_resize(self.feature_pyramid['P%d' % i],
                                                boxes=tf.stack([unit_xmin, unit_ymin, unit_xmax, unit_ymax], axis=1),
                                                box_ind=tf.zeros(shape=[tf.shape(level_rpn_boxes)[0], ], dtype=tf.int32),
                                                crop_size=[14, 14])

                level_cropped_rois = slim.max_pool2d(level_cropped_rois, [2, 2], stride=2)
                level_cropped_rois_list.append(level_cropped_rois)

            level_cropped_rois = tf.concat(level_cropped_rois_list, axis=0)
            level_rpn_boxes = tf.concat(level_rpn_boxes_list, axis=0)
            return level_cropped_rois, level_rpn_boxes

        else:
            xmin, ymin, xmax, ymax = tf.unstack(self.rpn_boxes, axis=1)
            unit_ymin = ymin / img_h
            unit_xmin = xmin / img_w
            unit_ymax = ymax / img_h
            unit_xmax = xmax / img_w

            cropped_rois = tf.image.crop_and_resize(self.featuremap,
                                                    boxes=tf.stack([unit_ymin, unit_xmin, unit_ymax, unit_xmax], axis=1),
                                                    box_ind=tf.zeros(shape=[tf.shape(self.rpn_boxes)[0], ], dtype=tf.int32),
                                                    crop_size=[14, 14])

            cropped_rois = slim.max_pool2d(cropped_rois, [2, 2], stride=2)

            #cropped_rois = tl.layers.InputLayer(cropped_rois)
            #cropped_rois = tl.layers.MaxPool2d(cropped_rois, filter_size=(2, 2), strides=(2, 2))
            return cropped_rois, self.rpn_boxes


    def fast_rcnn_inference(self):
        '''
        roi_input = tl.layers.InputLayer(self.roi)
        roi_flatten = tl.layers.FlattenLayer(roi_input, name='roi_flatten')

        fc1 = tl.layers.DenseLayer(roi_flatten, 1024, act=tf.nn.relu, name='fast_rcnn_fc1')
        fc2 = tl.layers.DenseLayer(fc1, 1024, act=tf.nn.relu, name='fast_rcnn_fc2')

        fast_rcnn_scores = tl.layers.DenseLayer(fc2, self.class_num + 1, act=tf.nn.relu, name='fast_rcnn_scores')
        fast_rcnn_coordinate_t = tl.layers.DenseLayer(fc2, self.class_num * 4, act=tf.nn.relu, name='fast_rcnn_coordinate_t')
        '''

        with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfg.FEATURE_WEIGHT_DECAY)):
            roi_flatten = slim.flatten(self.roi)
            fc1 = slim.fully_connected(roi_flatten, 4096, scope='fc1')
            fc1 = slim.dropout(fc1, keep_prob=0.5, is_training=self.is_training, scope='dropout1')
            fc2 = slim.fully_connected(fc1, 4096, scope='fc2')
            fc2 = slim.dropout(fc2, keep_prob=0.5, is_training=self.is_training, scope='dropout2')
            fast_rcnn_scores = slim.fully_connected(fc2, self.class_num+1, activation_fn=None, scope='fast_rcnn_classifier')
            fast_rcnn_coordinate_t = slim.fully_connected(fc2, (self.class_num+1)*4, activation_fn=None, scope='fast_rcnn_regressor')

        return fast_rcnn_scores, fast_rcnn_coordinate_t


    def define_samples(self):

        gtboxes = tf.cast(self.gtboxes_and_label[:, :-1], tf.float32)
        iou_each_roi = iou.iou_calculate(self.rpn_boxes, gtboxes)

        iou_each_roi_max = tf.reduce_max(iou_each_roi, axis=1)
        sample_kind = tf.cast(tf.greater_equal(iou_each_roi_max, self.positive_threshold), tf.int32)

        maxmatch_gtboxes_index = tf.argmax(iou_each_roi, axis=1)

        gtbox_each_roi = tf.gather(gtboxes, maxmatch_gtboxes_index)

        labels = tf.cast(self.gtboxes_and_label[:, -1], tf.int32)
        label_each_roi = sample_kind * tf.gather(labels, maxmatch_gtboxes_index)

        return gtbox_each_roi, label_each_roi, sample_kind


    def get_minibatch(self):

        gtbox_each_roi, label_each_roi, sample_kind = self.define_samples()

        positive_index = tf.reshape(tf.where(tf.equal(sample_kind, 1)), [-1])
        positive_index = tf.random_shuffle(positive_index)
        positive_num = tf.minimum(tf.shape(positive_index)[0],
                                  tf.cast(self.minibatch_size * self.minibatch_positive_ratio, dtype=tf.int32))
        positive_minibatch_index = tf.slice(positive_index, begin=[0], size=[positive_num])

        negative_index = tf.reshape(tf.where(tf.equal(sample_kind, 0)), [-1])
        negative_index = tf.random_shuffle(negative_index)
        negative_num = tf.minimum(tf.shape(negative_index)[0],
                                  tf.cast(self.minibatch_size, tf.int32) - positive_num)
        negative_minibatch_index = tf.slice(negative_index, begin=[0], size=[negative_num])

        minibatch_index = tf.concat([positive_minibatch_index, negative_minibatch_index], axis=0)
        minibatch_index = tf.random_shuffle(minibatch_index)

        minibatch_labels = tf.gather(label_each_roi, minibatch_index)
        minibatch_gtboxes = tf.gather(gtbox_each_roi, minibatch_labels)

        return sample_kind, minibatch_index, minibatch_labels, minibatch_gtboxes



    def fast_rcnn_loss(self):

        sample_kind, minibatch_index, minibatch_labels, minibatch_gtboxes = self.get_minibatch()

        sample_kind = tf.gather(tf.cast(sample_kind, tf.float32), minibatch_index)
        minibatch_roi = tf.gather(self.rpn_boxes, minibatch_index)
        minibatch_scores = tf.gather(self.scores, minibatch_index)
        minibatch_coordinate_t = tf.gather(self.coordinate_t, minibatch_index)

        #minibatch_roi_img = draw_box_with_color(self.img_batch, minibatch_roi, 0)
        positive_roi_img = draw_box_with_color(self.img_batch, minibatch_roi * tf.expand_dims(sample_kind, 1), 0)
        #tf.summary.image('minibatch_roi', minibatch_roi_img)
        tf.summary.image('positive_roi', positive_roi_img)

        regression_vector = box_regression.boxes_to_t(minibatch_roi, minibatch_gtboxes)

        #regression_vector, inside_weight = tf.py_func(_get_train_t, inp=[regression_vector, minibatch_labels, self.class_num], Tout=[tf.float32, tf.float32])
        #regression_vector = tf.reshape(regression_vector, [-1, (self.class_num+1)*4])
        #inside_weight = tf.reshape(inside_weight, [-1, (self.class_num + 1) * 4])

        minibatch_coordinate_t = tf.py_func(_find_match_t, inp=[minibatch_coordinate_t, minibatch_labels], Tout=tf.float32)
        minibatch_coordinate_t = tf.reshape(minibatch_coordinate_t, [-1, 4])

        #classification_loss = tl.cost.cross_entropy(minibatch_scores, minibatch_labels, name='fast_rcnn_class_loss')
        classification_loss = slim.losses.sparse_softmax_cross_entropy(minibatch_scores, minibatch_labels)
        location_loss = cfg.FAST_RCNN_LOSS_LAMBDA * smooth_l1_loss(minibatch_coordinate_t, regression_vector, sample_kind)
        #location_loss = cfg.FAST_RCNN_LOSS_LAMBDA * smooth_l1_loss_(minibatch_coordinate_t, regression_vector, inside_weight)
        #location_loss = cfg.FAST_RCNN_LOSS_LAMBDA * modified_smooth_l1(1., minibatch_coordinate_t, regression_vector, inside_weight)
        #slim.losses.add_loss(location_loss)

        return classification_loss, location_loss


    def boxes_predict(self):
        '''
        predict_scores = tf.nn.softmax(self.scores)
        predict_class = tf.argmax(predict_scores, axis=1)
        predict_t = tf.py_func(_find_match_t, inp=[self.coordinate_t, predict_class], Tout=tf.float32)
        predict_t = tf.reshape(predict_t, [-1, 4])
        predict_boxes = box_regression.t_to_boxes(self.rpn_boxes, predict_t)

        nms_index = tf.image.non_max_suppression()

        show_index = tf.reshape(tf.where(tf.greater_equal(predict_scores, self.show_boxes_threshold)), [-1])
        show_boxes = tf.gather(predict_boxes, show_index)
        show_scores = tf.gather(predict_scores, show_index)
        show_class = tf.gather(predict_class, show_index)

        return show_boxes, show_scores, show_class

'''        
        rpn_boxes = tf.reshape(tf.tile(self.rpn_boxes, [1, self.class_num]), [-1, 4])
        coordinate_t = tf.reshape(self.coordinate_t[:, 4:], [-1, 4])

        predict_boxes = box_regression.t_to_boxes(rpn_boxes, coordinate_t)
        predict_boxes = boundary_process.boundary_process(predict_boxes, self.img_shape)
        predict_boxes = tf.reshape(predict_boxes, [-1, self.class_num * 4])

        predict_scores = tf.nn.softmax(self.scores)
        predict_class = tf.argmax(predict_scores, axis=1)

        foreground = tf.cast(tf.not_equal(predict_class, 0), tf.float32)
        predict_boxes = predict_boxes * tf.reshape(foreground, [-1, 1]) # list to col
        predict_scores = predict_scores * tf.reshape(foreground, [-1, 1])

        predict_boxes = tf.reshape(predict_boxes, [-1, self.class_num, 4])
        boxes_list = tf.unstack(predict_boxes, axis=1)

        scores_list = tf.unstack(predict_scores[:, 1:], axis=1)

        boxes_nms_list = []
        scores_nms_list = []
        class_nms_list = []

        for boxes_each_class, scores_each_class in zip(boxes_list, scores_list):

            index_each_class = tf.image.non_max_suppression(boxes_each_class, scores_each_class,
                                                            self.boxes_num_each_class, self.nms_threshold_each_class)
            boxes_nms_list.append(tf.gather(boxes_each_class, index_each_class))
            scores_nms_list.append(tf.gather(scores_each_class, index_each_class))
            class_nms_list.append(tf.gather(predict_class, index_each_class))

        final_predict_boxes = tf.concat(boxes_nms_list, axis=0)
        final_predict_scores = tf.concat(scores_nms_list, axis=0)
        final_predict_class = tf.concat(class_nms_list, axis=0)

        show_index = tf.reshape(tf.where(tf.greater_equal(final_predict_scores, self.show_boxes_threshold)), [-1])
        show_boxes = tf.gather(final_predict_boxes, show_index)
        show_scores = tf.gather(final_predict_scores, show_index)
        show_class = tf.gather(final_predict_class, show_index)

        return show_boxes, show_scores, show_class












