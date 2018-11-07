# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append('../')

import tensorflow as tf
import tensorlayer as tl
import tensorflow.contrib.slim as slim
import numpy as np

import os
import time
import cv2

from config import config as cfg
from input.read_tfrecord import next_batch
from input import preprocess
from lib.networks.feature_net import get_featuremap
from lib.rpn.rpn import RPN
from lib.fast_rcnn.fast_rcnn import FAST_RCNN
from lib.model.model import get_restorer
from lib.boxes.box_cv import draw_box_cv, draw_box_with_color

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_GROUP



class Solver:
    def __init__(self, pretrained_model=None):
        self.pretrained_model = pretrained_model


    def get_gtbox(self, label):
        if cfg.LABEL_HORIZEN:
            return tf.squeeze(label, 0)[:cfg.CLASS_NUM, :]
        '''
        if cfg.LABEL_INCLINED:
            label_yxhwt = tf.py_func(back_forward_convert,
                                           inp=[tf.squeeze(label, 0)],
                                           Tout=tf.float32)
            return get_horizen_minAreaRectangle(label_yxhwt)
        '''


    def train_model(self):
        with tf.Graph().as_default():
            with tf.name_scope('get_batch'):
                img_batch, gtboxes_and_label_batch = \
                    next_batch(dataset_name=cfg.DATASET_NAME,
                               batch_size=cfg.BATCH_SIZE,
                               shortside_len=cfg.SHORT_SIDE_LEN,
                               is_training=True)

                gtboxes_and_label = tf.cast(self.get_gtbox(gtboxes_and_label_batch), tf.float32)

                with tf.name_scope('draw_gtboxes'):
                    gtboxes = draw_box_with_color(img_batch, gtboxes_and_label[:, :-1], tf.shape(gtboxes_and_label)[0])
                    tf.summary.image('gtboxes', gtboxes)

            with tf.name_scope('feature_extract'):
                featuremap = get_featuremap(net_name=cfg.NET_NAME,
                                                            input=img_batch)

            with tf.name_scope('rpn'):
                rpn = RPN(img_batch, tf.shape(img_batch), gtboxes_and_label, featuremap, is_training=True)
                rpn_classification_loss, rpn_location_loss = rpn.rpn_loss()
                rpn_loss = rpn_classification_loss + rpn_location_loss
                rpn_scores, rpn_boxes = rpn.get_rpn()

                #rpn_img = draw_box_with_color(img_batch, rpn_boxes, 0)
                #tf.summary.image('rpn_boxes', rpn_img)

            with tf.name_scope('fast_rcnn'):
                fast_rcnn = FAST_RCNN(img_batch, tf.shape(img_batch), gtboxes_and_label, featuremap, rpn.feature_pyramid, rpn_scores, rpn_boxes, is_training=True)
                fast_rcnn_classification_loss, fast_rcnn_location_loss = fast_rcnn.fast_rcnn_loss()
                fast_rcnn_loss = fast_rcnn_classification_loss + fast_rcnn_location_loss
                show_boxes, show_scores, show_class = fast_rcnn.boxes_predict()

                show_boxes_img = draw_box_with_color(img_batch, show_boxes, 0)
                tf.summary.image('show_boxes', show_boxes_img)

            weight_loss = tf.reduce_sum(slim.losses.get_regularization_losses())
            total_loss = rpn_loss + fast_rcnn_classification_loss + weight_loss

            global_step = slim.get_or_create_global_step()
            learning_rate = tf.train.piecewise_constant(global_step,
                                                        boundaries=[np.int64(15000), np.int64(30000), np.int64(45000)],
                                                        values=[cfg.BASE_LR, cfg.BASE_LR / 10, cfg.BASE_LR / 100, cfg.BASE_LR / 1000])
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=cfg.MOMENTUM)
            #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=global_step)
            #train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)

            summary_path = os.path.join(cfg.SUMMARY_PATH, cfg.VERSION)
            if not os.path.isdir(summary_path):
                os.mkdir(summary_path)

            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('rpn_cla_loss', rpn_classification_loss)
            tf.summary.scalar('rpn_loc_loss', rpn_location_loss)
            tf.summary.scalar('rpn_total_loss', rpn_loss)
            tf.summary.scalar('fast_rcnn_cla_loss', fast_rcnn_classification_loss)
            tf.summary.scalar('fast_rcnn_loc_loss', fast_rcnn_location_loss)
            tf.summary.scalar('fast_rcnn_total_loss', fast_rcnn_loss)
            tf.summary.scalar('total_loss', total_loss)
            summary_op = tf.summary.merge_all()

            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            saver = tf.train.Saver(max_to_keep=10)

            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                sess.run(init_op)

                if self.pretrained_model:
                    restorer, restore_ckpt = get_restorer()
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)

                summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

                for step in range(cfg.MAX_ITERATION):
                    training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

                    _global_step, _learning_rate, _, _total_loss, \
                    _rpn_classification_loss, _rpn_location_loss, _rpn_loss, \
                    _fast_rcnn_classification_loss, _fast_rcnn_location_loss, _fast_rcnn_loss, \
                    = sess.run([global_step, learning_rate, train_op, total_loss,
                                rpn_classification_loss, rpn_location_loss, rpn_loss,
                                fast_rcnn_classification_loss, fast_rcnn_location_loss, fast_rcnn_loss])

                    if step % cfg.SNAPSHOT_STEP == 0:

                        print('''{}: iterations {}   learning_rate = {}
                                              rpn:   rpn_cla_loss: {}
                                                     rpn_loc_loss: {}
                                                     rpn_total_loss: {}
                                        fast_rcnn:   fast_rcnn_cla_loss: {}
                                                     fast_rcnn_loc_loss: {}
                                                     fast_rcnn_total_loss: {}
                                            total:   total_loss: {}'''
                              .format(training_time, _global_step, _learning_rate,
                                      _rpn_classification_loss, _rpn_location_loss, _rpn_loss,
                                      _fast_rcnn_classification_loss, _fast_rcnn_location_loss, _fast_rcnn_loss,
                                      _total_loss))

                        summary = sess.run(summary_op)
                        summary_writer.add_summary(summary, _global_step)
                        summary_writer.flush()

                    if (step > 0 and step % cfg.SAVE_MODEL_STEP == 0) or step == cfg.MAX_ITERATION - 1:

                        save_dir = os.path.join(cfg.SAVE_MODEL_PATH, cfg.VERSION)
                        if not os.path.isdir(save_dir):
                            os.mkdir(save_dir)
                        save_model = os.path.join(save_dir, 'model_iteration_{}'.format(_global_step))
                        saver.save(sess, save_model)
                        print('model_iteration_{} is saved'.format(_global_step))

                coord.request_stop()
                coord.join(threads)


    def get_imgs(self):

        img_name_list = os.listdir(cfg.INFERENCE_IMG_PATH)
        img_list = [cv2.imread(os.path.join(cfg.INFERENCE_IMG_PATH, img_name)) for img_name in img_name_list]
        return img_name_list, img_list

    def inference(self):
        with tf.Graph().as_default():

            img_place = tf.placeholder(shape=[None, None, 3], dtype=tf.int32)
            img_tensor = tf.cast(img_place, tf.float32) - tf.constant([103.939, 116.779, 123.68])
            img_batch = preprocess.short_side_resize_for_inference_data(img_tensor,
                                                                        target_shortside_len=cfg.SHORT_SIDE_LEN)

            with tf.name_scope('feature_extract'):
                featuremap = get_featuremap(net_name=cfg.NET_NAME,
                                            input=img_batch)

            with tf.name_scope('rpn'):
                rpn = RPN(img_batch, tf.shape(img_batch), None, featuremap, is_training=False)
                rpn_scores, rpn_boxes = rpn.get_rpn()

            with tf.name_scope('fast_rcnn'):
                fast_rcnn = FAST_RCNN(img_batch, tf.shape(img_batch), None, featuremap, rpn.feature_pyramid, rpn_scores, rpn_boxes, is_training=False)
                show_boxes, show_scores, show_class = fast_rcnn.boxes_predict()

            init_op = tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()
            )

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                sess.run(init_op)

                if self.pretrained_model:
                    restorer, restore_ckpt = get_restorer()
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')

                img_name_list, img_list = self.get_imgs()

                for i, img in enumerate(img_list):
                    _img_batch, _show_boxes, _show_scores, _show_class = sess.run(
                        [img_batch, show_boxes, show_scores, show_class], feed_dict={img_place: img})

                    img_show = np.squeeze(_img_batch)

                    img_show = draw_box_cv(img_show, _show_boxes, _show_class, _show_scores)

                    if not os.path.isdir(cfg.INFERENCE_RESULT_PATH):
                        os.mkdir(cfg.INFERENCE_RESULT_PATH)

                    cv2.imwrite(os.path.join(cfg.INFERENCE_RESULT_PATH, '{}.jpg'.format(i)), img_show)

                    '''
                    init_op = tf.group(
                        tf.global_variables_initializer(),
                        tf.local_variables_initializer()
                    )
                    sess = tf.Session()
                    sess.run(init_op)
                    print(sess.run( ,feed_dict={img_place: img}))
                    '''




if __name__ == '__main__':
    solver = Solver(pretrained_model=True)
    solver.train_model()
    #solver.inference()
























