# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
from input.label_dict import LABEl_NAME_MAP


def draw_box_cv(img, boxes, labels, scores):
    img = img + np.array([103.939, 116.779, 123.68])
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)
    img = np.array(img, np.float32)
    img = np.array(img*255/np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        label = labels[i]
        if label != 0:
            num_of_object += 1
            #color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=(0, 0, 255),#color,
                          thickness=2)

            category = LABEl_NAME_MAP[label]

            if scores is not None:
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin+150, ymin+15),
                              color=(255, 0, 0),#color,
                              thickness=-1)
                cv2.putText(img,
                            text=category+": "+str(scores[i]),
                            org=(xmin, ymin+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(255, 255, 255))#(color[1], color[2], color[0]))
            else:
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin + 40, ymin + 15),
                              color=(255, 0, 0),#color,
                              thickness=-1)
                cv2.putText(img,
                            text=category,
                            org=(xmin, ymin + 10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(255, 255, 255))#(color[1], color[2], color[0]))
    '''            
    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=1,
                color=(255, 0, 0))
    '''
    return img


def draw_box_with_color(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array([103.939, 116.779, 123.68])
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        # img = np.transpose(img, [2, 1, 0])
        img = img[:, :, -1::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    # color = tf.constant([0, 0, 255])
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])
    #img_tensor_with_boxes = draw_box_cv(img_tensor, boxes, text)

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes