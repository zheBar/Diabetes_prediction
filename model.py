import tensorflow as tf

from tensorflow.contrib import slim


class InceptionNet(object):

    def __init__(self):
        print('init InceptionNet')
        # return self.model

    def conv2d_bn(self, input_data, kernel_num, filter_size=[2, 2], stride=1, padding='SAME'):
        res = slim.conv2d(
            input_data,
            kernel_num,
            filter_size,
            stride=stride,
            padding=padding,
            activation_fn=None)
        res = slim.batch_norm(res)
        return tf.nn.relu(res)

    def _stem(self, x):
        # x.shape=[batch_num, 1, 40, 1]
        with tf.variable_scope('stem'):
            x = self.conv2d_bn(x, 32, filter_size=[1, 3])  # shape=[batch_num, 1, 40, 32]

            x1 = self.conv2d_bn(x, 64, [1, 2])  # shape=[batch_num, 1, 40, 64])

            x2 = self.conv2d_bn(x, 64, [1, 3])  # shape=[batch_num, 1, 40, 64]
            return tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 40, 128]

    def _inception_1(self, x):
        # x.shape=[batch_num, 1, 40 , 96]
        with tf.variable_scope('inception_1'):
            x1 = self.conv2d_bn(x, 64, [1, 5])  # shape=[batch_num, 1, 40, 64]

            x2 = self.conv2d_bn(x, 64, [1, 3])  # shape=[batch_num, 1, 40, 64]

            res = tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 40, 128]
            res = tf.add(x, res)  # shape=[batch_num, 1, 40, 128]
        return res

    def _inception_2(self, x):
        # shape=[batch_num, 1, 40, 128]
        with tf.variable_scope('inception_2'):
            x1 = self.conv2d_bn(x, 128, [1, 10])  # shape=[batch_num, 1, 40, 128]

            x2 = self.conv2d_bn(x, 128, [1, 10])  # shape=[batch_num, 1, 10, 128]

            res = tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 40, 256]
        return res

    def _inception_3(self, x):
        # shape=[batch_num, 1, 40, 288]
        with tf.variable_scope('inception_3'):
            x = slim.avg_pool2d(x, [1, 2], stride=[1, 2], padding='SAME')  # shape=[batch_num, 1, 20, 256]

            x1 = self.conv2d_bn(x, 48, [1, 5])  # shape=[batch_num, 1, 20, 48]

            x2 = self.conv2d_bn(x, 42, [1, 5])  # shape=[batch_num, 1, 20, 42]

            res = tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 20, 96]
            res = self.conv2d_bn(res, 288, [1, 1])  # shape=[batch_num, 1, 20, 288]
        return res

    def _inception_4(self, x):
        # shape=[batch_num, 1, 20, 288]
        with tf.variable_scope('inception_4'):
            x1 = self.conv2d_bn(x, 48, [1, 10])  # shape=[batch_num, 1, 20, 48]

            x2 = self.conv2d_bn(x, 42, [1, 10])  # shape=[batch_num, 1, 20, 42]

            res = tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 20, 96]
            res = self.conv2d_bn(res, 288, [1, 1])  # shape=[batch_num, 1, 20, 288]
            res = tf.add(x, res)  # shape=[batch_num, 1, 20, 288]
        return res

    def _inception_5(self, x):
        # shape=[batch_num, 1, 20, 288]
        with tf.variable_scope('inception_5'):
            x1 = self.conv2d_bn(x, 128, [1, 5])  # shape=[batch_num, 1, 20, 128]

            x2 = self.conv2d_bn(x, 128, [1, 5])  # shape=[batch_num, 1, 20, 128]

            res = tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 20, 256]
        return res

    def _end_block(self, x):
        # shape=[batch_num, 1, 20, 256]
        with tf.variable_scope('end_block'):
            x1 = self.conv2d_bn(x, 128, [1, 3])  # shape=[batch_num, 1, 20, 128]

            x2 = self.conv2d_bn(x, 128, [1, 4])  # shape=[batch_num, 1, 20, 128]
            res = tf.concat([x1, x2], 3)  # shape=[batch_num, 1, 20, 256]
            res = slim.avg_pool2d(res, [1, 2], stride=[1, 2], padding='SAME')  # shape=[batch_num, 1, 10, 288]
            res = slim.flatten(res)  # shape=[batch_num, 288]

            logits = slim.fully_connected(res, 1, activation_fn=None)  # shape=[batch_num, 1]
        return logits

    def get_model(self, input_data):
        with tf.variable_scope('inceptionNet'):
            res = self._stem(input_data)
            res = self._inception_1(res)
            res = self._inception_2(res)
            res = self._inception_3(res)
            res = self._inception_4(res)
            res = self._inception_5(res)
            self.model = self._end_block(res)
        return self.model