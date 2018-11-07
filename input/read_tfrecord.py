# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import os
from input import preprocess
from config import config as cfg


def read_single_example_and_decode(filename_queue):

    tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    reader = tf.TFRecordReader(options=tfrecord_options)

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=serialized_example,
        features={
            #'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    #img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])

    #num_objects = tf.cast(features['num_objects'], tf.int32)
    return img, gtboxes_and_label#, num_objects


def read_and_prepocess_single_img(filename_queue, shortside_len, is_training):

    img, gtboxes_and_label = read_single_example_and_decode(filename_queue)

    # img = tf.image.per_image_standardization(img)
    img = tf.cast(img, tf.float32)
    img = img - tf.constant([103.939, 116.779, 123.68])
    if is_training:
        img, gtboxes_and_label = preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                              target_shortside_len=shortside_len)
        img, gtboxes_and_label = preprocess.random_flip_left_right(img_tensor=img, gtboxes_and_label=gtboxes_and_label)

    else:
        img, gtboxes_and_label = preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                              target_shortside_len=shortside_len)

    return img, gtboxes_and_label



def next_batch(dataset_name, batch_size, shortside_len, is_training):
    #if dataset_name not in ['ship', 'spacenet', 'pascal', 'coco']:
    #    raise ValueError('dataSet name must be in pascal or coco')

    if is_training:
        pattern = os.path.join('/home/aemc/my/faster/data/tfrecords', 'train_' + dataset_name + '.tfrecord')
        filename_queue = tf.train.string_input_producer([pattern])
    else:
        pattern = os.path.join('/home/aemc/my/faster/data/tfrecords', 'test_' + dataset_name + '.tfrecord')
        filename_queue = tf.train.string_input_producer([pattern], num_epochs=1)

    print('tfrecord path is -->', os.path.abspath(pattern))
    #filename_tensorlist = tf.train.match_filenames_once(pattern)
    #filename_queue = tf.train.string_input_producer([pattern])

    img, gtboxes_and_label = read_and_prepocess_single_img(filename_queue, shortside_len,
                                                                              is_training=is_training)
    '''
    sv = tf.train.Supervisor()
    with sv.managed_session() as sess:
        a = sess.run(gtboxes_and_label)
        print(a)
    '''

    img_batch, gtboxes_and_label_batch = \
        tf.train.batch(
                       [img, gtboxes_and_label],
                       batch_size=batch_size,
                       capacity=320,
                       num_threads=16,
                       dynamic_pad=True
                      )

    return img_batch, gtboxes_and_label_batch

