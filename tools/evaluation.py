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
import pickle

from config import config as cfg
from input.read_tfrecord import next_batch
from input import preprocess
from lib.networks.feature_net import get_featuremap
from lib.rpn.rpn import RPN
from lib.fast_rcnn.fast_rcnn import FAST_RCNN
from lib.model.model import get_restorer
from lib.boxes.box_cv import draw_box_cv, draw_box_with_color
from input.label_dict import *

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_GROUP



def make_dict_packle(_gtboxes_and_label, _fast_rcnn_decode_boxes, _fast_rcnn_score, _detection_category):

    gtbox_list = []
    predict_list = []

    for j, box in enumerate(_gtboxes_and_label):
        bbox_dict = {}
        bbox_dict['bbox'] = np.array(_gtboxes_and_label[j, :-1], np.float64)
        bbox_dict['name'] = LABEl_NAME_MAP[int(_gtboxes_and_label[j, -1])]
        gtbox_list.append(bbox_dict)

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue
        else:
            temp_dict = {}
            temp_dict['name'] = label

            ind = np.where(_detection_category == NAME_LABEL_MAP[label])[0]
            temp_boxes = _fast_rcnn_decode_boxes[ind]
            temp_score = np.reshape(_fast_rcnn_score[ind], [-1, 1])
            temp_dict['bbox'] = np.array(np.concatenate([temp_boxes, temp_score], axis=1), np.float64)
            predict_list.append(temp_dict)
    return gtbox_list, predict_list

def save_result():
    with tf.Graph().as_default():
        with tf.name_scope('get_batch'):
            img_batch, gtboxes_and_label_batch = \
                next_batch(dataset_name=cfg.DATASET_NAME,
                           batch_size=cfg.BATCH_SIZE,
                           shortside_len=cfg.SHORT_SIDE_LEN,
                           is_training=False)

            gtboxes_and_label = tf.cast(tf.squeeze(gtboxes_and_label_batch, 0)[:cfg.CLASS_NUM, :], tf.float32)

        with tf.name_scope('feature_extract'):
            featuremap = get_featuremap(net_name=cfg.NET_NAME,
                                        input=img_batch)

        with tf.name_scope('rpn'):
            rpn = RPN(img_batch, tf.shape(img_batch), gtboxes_and_label, featuremap, is_training=False)
            rpn_scores, rpn_boxes = rpn.get_rpn()

        with tf.name_scope('fast_rcnn'):
            fast_rcnn = FAST_RCNN(img_batch, tf.shape(img_batch), gtboxes_and_label, featuremap, rpn.feature_pyramid,
                                  rpn_scores, rpn_boxes, is_training=False)
            show_boxes, show_scores, show_class = fast_rcnn.boxes_predict()

        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init_op)

            restorer, restore_ckpt = get_restorer()
            restorer.restore(sess, restore_ckpt)
            print('restore model')

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            gtboxes_dict = {}
            predict_dict = {}
            i = 0

            try:
                while(True):

                    _gtboxes_and_label, _show_boxes, _show_scores, _show_class = \
                        sess.run([gtboxes_and_label, show_boxes, show_scores, show_class])

                    gtboxes_dict[str(i)] = []
                    predict_dict[str(i)] = []

                    gtbox_list, predict_list = make_dict_packle(_gtboxes_and_label, _show_boxes, _show_scores, _show_class)

                    gtboxes_dict[str(i)].extend(gtbox_list)
                    predict_dict[str(i)].extend(predict_list)

                    i += 1

            except tf.errors.OutOfRangeError:
                print('test done')

            finally:
                coord.request_stop()
                coord.join(threads)

            fw1 = open('gtboxes_dict.pkl', 'wb')
            fw2 = open('predict_dict.pkl', 'wb')
            pickle.dump(gtboxes_dict, fw1)
            pickle.dump(predict_dict, fw2)
            fw1.close()
            fw2.close()

def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """

    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_single_label_dict(predict_dict, gtboxes_dict, label):
    rboxes = {}
    gboxes = {}
    rbox_images = list(predict_dict.keys())
    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        for pre_box in predict_dict[rbox_image]:
            if pre_box['name'] == label and len(pre_box['bbox']) != 0:
                rboxes[rbox_image] = [pre_box]

                gboxes[rbox_image] = []

                for gt_box in gtboxes_dict[rbox_image]:
                    if gt_box['name'] == label:
                        gboxes[rbox_image].append(gt_box)
    return rboxes, gboxes


def eval(rboxes, gboxes, iou_th, use_07_metric):
    rbox_images = list(rboxes.keys())
    fp = np.zeros(len(rbox_images))
    tp = np.zeros(len(rbox_images))
    box_num = 0

    for i in range(len(rbox_images)):
        rbox_image = rbox_images[i]
        if len(rboxes[rbox_image][0]['bbox']) > 0:

            rbox_lists = np.array(rboxes[rbox_image][0]['bbox'])
            if len(gboxes[rbox_image]) > 0:
                gbox_list = np.array([obj['bbox'] for obj in gboxes[rbox_image]])
                box_num = box_num + len(gbox_list)
                gbox_list = np.concatenate((gbox_list, np.zeros((np.shape(gbox_list)[0], 1))), axis=1)
                confidence = rbox_lists[:, -1]
                box_index = np.argsort(-confidence)

                rbox_lists = rbox_lists[box_index, :]
                for rbox_list in rbox_lists:
                    ixmin = np.maximum(gbox_list[:, 0], rbox_list[0])
                    iymin = np.maximum(gbox_list[:, 1], rbox_list[1])
                    ixmax = np.minimum(gbox_list[:, 2], rbox_list[2])
                    iymax = np.minimum(gbox_list[:, 3], rbox_list[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih
                    # union
                    uni = ((rbox_list[2] - rbox_list[0] + 1.) * (rbox_list[3] - rbox_list[1] + 1.) +
                           (gbox_list[:, 2] - gbox_list[:, 0] + 1.) *
                           (gbox_list[:, 3] - gbox_list[:, 1] + 1.) - inters)
                    overlaps = inters / uni

                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
                    if ovmax > iou_th:
                        if gbox_list[jmax, -1] == 0:
                            tp[i] += 1
                            gbox_list[jmax, -1] = 1
                        else:
                            fp[i] += 1
                    else:
                        fp[i] += 1
            else:
                fp[i] += len(rboxes[rbox_image][0]['bbox'])
        else:
            continue
    rec = np.zeros(len(rbox_images))
    prec = np.zeros(len(rbox_images))
    if box_num == 0:
        for i in range(len(fp)):
            if fp[i] != 0:
                prec[i] = 0
            else:
                prec[i] = 1

    else:

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        rec = tp / box_num
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap, box_num

if __name__ == '__main__':

    save_result()

    fr1 = open('gtboxes_dict.pkl', 'rb')
    fr2 = open('predict_dict.pkl', 'rb')
    gtboxes_dict = pickle.load(fr1)
    predict_dict = pickle.load(fr2)

    R, P, AP, F, num = [], [], [], [], []
    R1, P1, AP1, F1, num1 = [], [], [], [], []

    for label in NAME_LABEL_MAP.keys():
        if label == 'back_ground':
            continue

        rboxes, gboxes = get_single_label_dict(predict_dict, gtboxes_dict, label)

        rec, prec, ap, box_num = eval(rboxes, gboxes, 0.5, False)

        recall = rec[-1]
        precision = prec[-1]
        F_measure = (2 * precision * recall) / (recall + precision)
        print('\n{}\tR:{:.2%}\tP:{:.2%}\tap:{:.2%}\tF:{:.2%}'.format(label, recall, precision, ap, F_measure))
        R.append(recall)
        P.append(precision)
        AP.append(ap)
        F.append(F_measure)
        num.append(box_num)

    R = np.array(R)
    P = np.array(P)
    AP = np.array(AP)
    F = np.array(F)
    num = np.array(num)
    weights = num / np.sum(num)
    Recall = np.sum(R * weights)
    Precision = np.sum(P * weights)
    mAP = np.sum(AP * weights)
    F_measure = np.sum(F * weights)

    print('\n{}\tR:{:.2%}\tP:{:.2%}\tmAP:{:.2%}\tF:{:.2%}'.format('horizontal standard', Recall, Precision, mAP, F_measure))

    fr1.close()
    fr2.close()







