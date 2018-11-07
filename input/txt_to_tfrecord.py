# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
from libs.configs import cfgs
import numpy as np
import tensorflow as tf
import glob
import cv2
#from libs.label_name_dict.label_dict import *
from help_utils.tools import *

tf.app.flags.DEFINE_string('VOC_dir', cfgs.ROOT_PATH + '/data/VOCdevkit_test/', 'Voc dir')
tf.app.flags.DEFINE_string('txt_dir', 'Annotations1', 'txt dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages1', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'hooks_test1', 'save name')
tf.app.flags.DEFINE_string('save_dir', cfgs.ROOT_PATH + '/data/tfrecords', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_txt(txt_path):
    img_height = 1280
    img_width = 1920
    box_list = []
    txt_file = open(txt_path, 'r')
    for line in txt_file.readlines():
        tmp_box = line.strip().split(',')
        #for i in xrange(8):
        #    tmp_box[i] = int(tmp_box[i])
        tmp_box[-1] = 1
        box_list.append(tmp_box)
    gtbox_label = np.array(box_list, dtype=np.int32)
    #print(gtbox_label)
    return img_height, img_width, gtbox_label

def convert_pascal_to_tfrecord():
    txt_path = FLAGS.VOC_dir + FLAGS.txt_dir
    image_path = FLAGS.VOC_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + '/' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)

    for count, txt in enumerate(glob.glob(txt_path + '/*.txt')):
        # to avoid path error in different development platform
        txt = txt.replace('\\', '/')

        img_name = txt.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue

        img_height, img_width, gtbox_label = read_txt(txt)

        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)
        '''
        hu = img.tostring()
        hu1 = img.tobytes()
        li = gtbox_label.tostring()
        li1 = np.fromstring(li)
        hoo = gtbox_label.shape[0]

        print(gtbox_label)
        print(img)
        '''
        feature = tf.train.Features(feature={
            # do not need encode() in linux
            # 'img_name': _bytes_feature(img_name.encode()),
            'img_name': _bytes_feature(img_name),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(txt_path + '/*.txt')))

    print('\nConversion is complete!')


if __name__ == '__main__':
    convert_pascal_to_tfrecord()
