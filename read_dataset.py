# -*- coding:utf-8 -*- 
# 读取tfrecords文件
# Author：FengZhijin

import tensorflow as tf

from util.tfrecord_tools import TFRecord
from tensorflow.contrib import slim

train_path = "./data/tfrecord/train/*.tfrecord"
validation_path = "./data/tfrecord/validation/*.tfrecord"


def read_data(cate="train", batch_size=64, batch_fun="batch"):
    tfr = TFRecord({'data': [float], 'label': [float]})
    if str(cate) == "train":
        example = tfr.reader(train_path)
    elif str(cate) == "validation":
        example = tfr.reader(validation_path)
    img = example['data']
    print(img)
    img = tf.reshape(img, [1, 40, 1])
    lab = example['label']
    if batch_fun == "batch":
        batch_image, batch_label = tf.train.batch(
            [img, lab], batch_size, num_threads=1, capacity=batch_size*4)
    elif batch_fun == "shuffle_batch":
        batch_image, batch_label = tf.train.shuffle_batch(
            [img, lab], batch_size, capacity=batch_size*4,
            min_after_dequeue=batch_size*2, num_threads=1)
    return batch_image, batch_label


if __name__ == "__main__":
    print(read_data('train'))