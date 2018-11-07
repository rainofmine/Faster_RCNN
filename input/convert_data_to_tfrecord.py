# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
from config import config as cfg
import numpy as np
import tensorflow as tf
import glob
import cv2
import math
from input.label_dict import *


voc_dir = cfg.ROOT_PATH + '/data/test_data/'
xml_dir = 'Annotations'
image_dir = 'JPEGImages'
save_name = 'test_hooks'
save_dir = cfg.ROOT_PATH + '/data/tfrecords/'
img_format = '.jpg'

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord():
    xml_path = voc_dir + xml_dir
    image_path = voc_dir + image_dir
    save_path = save_dir + save_name + '.tfrecord'
    mkdir(save_dir)

    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)

    for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')

        img_name = xml.split('/')[-1].split('.')[0] + img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)

        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)

        feature = tf.train.Features(feature={
            # do not need encode() in linux
            # 'img_name': _bytes_feature(img_name.encode()),
            #'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    print('\nConversion is complete!')


if __name__ == '__main__':

    convert_pascal_to_tfrecord()
