# coding: utf-8

import tensorflow as tf
import xlrd
import re
import numpy as np
import csv
import read_data1 as rd


classes = {'A': 0.0, 'B': 1.0, 'C': 2.0, 'D': 3.0, 'E': 4.0, 'F': 5.0,
           'G': 6.0, 'H': 7.0, 'I': 8.0, 'J': 9.0, 'K': 10.0, 'L': 11.0,
           'M': 12.0, 'N': 13.0, 'O': 14.0, 'P': 15.0, 'Q': 16.0,
           'R': 17.0, 'S': 18.0, 'T': 19.0, 'U': 20.0, 'V': 21.0,
           'W': 22.0,  'X': 23.0, 'Y': 24.0, 'Z': 25.0, 'E0': 26.0,
           'YX1': 27.0, 'N0': 28.0, 'XY1': 29.0}


def read_xlsx():
    workbook = xlrd.open_workbook('./训练_01.xlsx')
    booksheet = workbook.sheet_by_name('default')
    p = list()
    for row in range(booksheet.nrows):
        row_data = []
        for col in range(booksheet.ncols):
            cel = booksheet.cell(row, col)
            val = cel.value
            try:
                val = cel.value
                val = re.sub(r'\s+', '', val)
            except:
                pass
            if val in classes:
                val = classes[val]
            elif val == '':
                val = 0.0
            row_data.append(val)
        p.append(row_data)
    return p


def min_mine(data):
    min_ = data[0]
    for i in data:
        if i != 0.0:
            min_ = min(min_, i)
    return min_


def data_prep():
    ls = read_xlsx()
    rows_number = len(ls)
    cols_number = len(ls[0])
    images = list()
    for i in range(501, rows_number):
        images.append(ls[i][1:cols_number - 1])
    rows_number = len(images)
    cols_number = len(images[0])
    images = np.array(images, dtype=float)
    for j in range(cols_number):
        max_ = max(images[:, j])
        min_ = min_mine(images[:, j])
        if (max_ - min_) == 0.0:
            images[:, j] = float(0.0)
        else:
            images[:, j] = (images[:, j] - min_) / (max_ - min_)
    print(len(images))
    print(len(images[0]))
    return images


def get_data():
    image_data = data_prep()
    return image_data.reshape(-1, 1, 8027, 1)


def init_weight(shape):
    weights = tf.Variable(tf.random_normal(shape, stddev=0.01))
    return weights


def y_handle(index):
    for i in range(len(index)):
        index_ = np.zeros((1, 12, 11), dtype=float)
        for j in range(13):
            xiabiao = 10
            if j == 0:
                xiabiao = int(str(index[i])[j])
                index_[0, 0, xiabiao] = 1.0
            elif j == 1:
                continue
            elif j < len(str(index[i])):
                xiabiao = int(str(index[i])[j])
                index_[0, j-1, xiabiao] = 1.0
        if i == 0:
            out_index = np.array(index_)
        else:
            out_index = np.concatenate((out_index, index_), axis=0)
    return out_index


def y_reduction(index):
    index_ = list()
    for i in range(len(index)):
        xiabiao = np.argmax(index[i], axis=1)
        out_index = ''
        for j in range(len(xiabiao)):
            if xiabiao[j] != 10:
                out_index = out_index + str(xiabiao[j])
        index_.append(int(out_index))
    return np.array(index_)


def loss_func(py_x, Y):
    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 0], labels=Y[:, 0]))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 1], labels=Y[:, 1]))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 2], labels=Y[:, 2]))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 3], labels=Y[:, 3]))
    loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 4], labels=Y[:, 4]))
    loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 5], labels=Y[:, 5]))
    loss6 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 6], labels=Y[:, 6]))
    loss7 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 7], labels=Y[:, 7]))
    loss8 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 8], labels=Y[:, 8]))
    loss9 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 9], labels=Y[:, 9]))
    loss10 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 10], labels=Y[:, 10]))
    loss11 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x[:, 11], labels=Y[:, 11]))
    loss = tf.reduce_sum([loss0, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11])
    return loss


def get_weights_bases(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1]], stddev=0.01))
    tf.add_to_collection('losses_weight', tf.contrib.layers.l2_regularizer(0.001)(weights))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[1]]))
    tf.add_to_collection('losses_base', tf.contrib.layers.l2_regularizer(0.001)(bases))
    return weights, bases


def get_weights_bases_wout(shape):
    weights = tf.Variable(tf.truncated_normal([shape[0], shape[1], shape[2]], stddev=0.01))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.001)(weights))
    bases = tf.Variable(tf.constant(0.1, shape=[shape[0], shape[2]]))
    tf.add_to_collection('losses_base', tf.contrib.layers.l2_regularizer(0.001)(bases))
    return weights, bases


def model(X, w, w1, w2, w3, w4, w5, w6, w7, b7, w8, b8, p_keep_conv, p_keep_within):
    la = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    l = tf.nn.avg_pool(la, ksize=[1, 1, 25, 1], strides=[1, 1, 5, 1], padding='VALID')
    l = tf.nn.dropout(l, p_keep_conv)
    # shape = [?, 1, 1601, 32]

    l1a = tf.nn.relu(tf.nn.conv2d(l, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.avg_pool(l1a, ksize=[1, 1, 25, 1], strides=[1, 1, 5, 1], padding='VALID')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    # shape = [?, 1, 316, 64]

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.avg_pool(l2a, ksize=[1, 1, 5, 1], strides=[1, 1, 5, 1], padding='VALID')
    l2 = tf.nn.dropout(l2, p_keep_within)
    # shape = [?, 1, 63, 64]

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.avg_pool(l3a, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_within)
    # shape = [?, 1, 21, 64]

    l4a = tf.nn.relu(tf.nn.conv2d(l3, w4, strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.avg_pool(l4a, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l4 = tf.nn.dropout(l4, p_keep_within)
    # shape = [?, 1, 21, 64]

    l5a_0 = tf.nn.relu(tf.nn.conv2d(l4, w5[0], strides=[1, 1, 1, 1], padding='SAME'))
    l5_0 = tf.nn.avg_pool(l5a_0, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_0 = tf.nn.dropout(l5_0, p_keep_within)
    l5a_1 = tf.nn.relu(tf.nn.conv2d(l4, w5[1], strides=[1, 1, 1, 1], padding='SAME'))
    l5_1 = tf.nn.avg_pool(l5a_1, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_1 = tf.nn.dropout(l5_1, p_keep_within)
    l5a_2 = tf.nn.relu(tf.nn.conv2d(l4, w5[2], strides=[1, 1, 1, 1], padding='SAME'))
    l5_2 = tf.nn.avg_pool(l5a_2, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_2 = tf.nn.dropout(l5_2, p_keep_within)
    l5a_3 = tf.nn.relu(tf.nn.conv2d(l4, w5[3], strides=[1, 1, 1, 1], padding='SAME'))
    l5_3 = tf.nn.avg_pool(l5a_3, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_3 = tf.nn.dropout(l5_3, p_keep_within)
    l5a_4 = tf.nn.relu(tf.nn.conv2d(l4, w5[4], strides=[1, 1, 1, 1], padding='SAME'))
    l5_4 = tf.nn.avg_pool(l5a_4, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_4 = tf.nn.dropout(l5_4, p_keep_within)
    l5a_5 = tf.nn.relu(tf.nn.conv2d(l4, w5[5], strides=[1, 1, 1, 1], padding='SAME'))
    l5_5 = tf.nn.avg_pool(l5a_5, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_5 = tf.nn.dropout(l5_5, p_keep_within)
    l5a_6 = tf.nn.relu(tf.nn.conv2d(l4, w5[6], strides=[1, 1, 1, 1], padding='SAME'))
    l5_6 = tf.nn.avg_pool(l5a_6, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_6 = tf.nn.dropout(l5_6, p_keep_within)
    l5a_7 = tf.nn.relu(tf.nn.conv2d(l4, w5[7], strides=[1, 1, 1, 1], padding='SAME'))
    l5_7 = tf.nn.avg_pool(l5a_7, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_7 = tf.nn.dropout(l5_7, p_keep_within)
    l5a_8 = tf.nn.relu(tf.nn.conv2d(l4, w5[8], strides=[1, 1, 1, 1], padding='SAME'))
    l5_8 = tf.nn.avg_pool(l5a_8, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_8 = tf.nn.dropout(l5_8, p_keep_within)
    l5a_9 = tf.nn.relu(tf.nn.conv2d(l4, w5[9], strides=[1, 1, 1, 1], padding='SAME'))
    l5_9 = tf.nn.avg_pool(l5a_9, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_9 = tf.nn.dropout(l5_9, p_keep_within)
    l5a_10 = tf.nn.relu(tf.nn.conv2d(l4, w5[10], strides=[1, 1, 1, 1], padding='SAME'))
    l5_10 = tf.nn.avg_pool(l5a_10, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_10 = tf.nn.dropout(l5_10, p_keep_within)
    l5a_11 = tf.nn.relu(tf.nn.conv2d(l4, w5[11], strides=[1, 1, 1, 1], padding='SAME'))
    l5_11 = tf.nn.avg_pool(l5a_11, ksize=[1, 1, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
    l5_11 = tf.nn.dropout(l5_11, p_keep_within)

    l6a_0 = tf.nn.relu(tf.nn.conv2d(l5_0, w6[0], strides=[1, 1, 1, 1], padding='SAME'))
    l6_0 = tf.reshape(l6a_0, [-1, w7.get_shape().as_list()[1]])
    l6_0 = tf.nn.dropout(l6_0, p_keep_within)
    l6a_1 = tf.nn.relu(tf.nn.conv2d(l5_1, w6[1], strides=[1, 1, 1, 1], padding='SAME'))
    l6_1 = tf.reshape(l6a_1, [-1, w7.get_shape().as_list()[1]])
    l6_1 = tf.nn.dropout(l6_1, p_keep_within)
    l6a_2 = tf.nn.relu(tf.nn.conv2d(l5_2, w6[2], strides=[1, 1, 1, 1], padding='SAME'))
    l6_2 = tf.reshape(l6a_2, [-1, w7.get_shape().as_list()[1]])
    l6_2 = tf.nn.dropout(l6_2, p_keep_within)
    l6a_3 = tf.nn.relu(tf.nn.conv2d(l5_3, w6[3], strides=[1, 1, 1, 1], padding='SAME'))
    l6_3 = tf.reshape(l6a_3, [-1, w7.get_shape().as_list()[1]])
    l6_3 = tf.nn.dropout(l6_3, p_keep_within)
    l6a_4 = tf.nn.relu(tf.nn.conv2d(l5_4, w6[4], strides=[1, 1, 1, 1], padding='SAME'))
    l6_4 = tf.reshape(l6a_4, [-1, w7.get_shape().as_list()[1]])
    l6_4 = tf.nn.dropout(l6_4, p_keep_within)
    l6a_5 = tf.nn.relu(tf.nn.conv2d(l5_5, w6[5], strides=[1, 1, 1, 1], padding='SAME'))
    l6_5 = tf.reshape(l6a_5, [-1, w7.get_shape().as_list()[1]])
    l6_5 = tf.nn.dropout(l6_5, p_keep_within)
    l6a_6 = tf.nn.relu(tf.nn.conv2d(l5_6, w6[6], strides=[1, 1, 1, 1], padding='SAME'))
    l6_6 = tf.reshape(l6a_6, [-1, w7.get_shape().as_list()[1]])
    l6_6 = tf.nn.dropout(l6_6, p_keep_within)
    l6a_7 = tf.nn.relu(tf.nn.conv2d(l5_7, w6[7], strides=[1, 1, 1, 1], padding='SAME'))
    l6_7 = tf.reshape(l6a_7, [-1, w7.get_shape().as_list()[1]])
    l6_7 = tf.nn.dropout(l6_7, p_keep_within)
    l6a_8 = tf.nn.relu(tf.nn.conv2d(l5_8, w6[8], strides=[1, 1, 1, 1], padding='SAME'))
    l6_8 = tf.reshape(l6a_8, [-1, w7.get_shape().as_list()[1]])
    l6_8 = tf.nn.dropout(l6_8, p_keep_within)
    l6a_9 = tf.nn.relu(tf.nn.conv2d(l5_9, w6[9], strides=[1, 1, 1, 1], padding='SAME'))
    l6_9 = tf.reshape(l6a_9, [-1, w7.get_shape().as_list()[1]])
    l6_9 = tf.nn.dropout(l6_9, p_keep_within)
    l6a_10 = tf.nn.relu(tf.nn.conv2d(l5_10, w6[10], strides=[1, 1, 1, 1], padding='SAME'))
    l6_10 = tf.reshape(l6a_10, [-1, w7.get_shape().as_list()[1]])
    l6_10 = tf.nn.dropout(l6_10, p_keep_within)
    l6a_11 = tf.nn.relu(tf.nn.conv2d(l5_11, w6[11], strides=[1, 1, 1, 1], padding='SAME'))
    l6_11 = tf.reshape(l6a_11, [-1, w7.get_shape().as_list()[1]])
    l6_11 = tf.nn.dropout(l6_11, p_keep_within)

    l7_0 = tf.nn.relu(tf.matmul(l6_0, w7[0]) + b7[0])
    l7_0 = tf.nn.dropout(l7_0, p_keep_conv)
    l7_1 = tf.nn.relu(tf.matmul(l6_1, w7[1]) + b7[1])
    l7_1 = tf.nn.dropout(l7_1, p_keep_conv)
    l7_2 = tf.nn.relu(tf.matmul(l6_2, w7[2]) + b7[2])
    l7_2 = tf.nn.dropout(l7_2, p_keep_conv)
    l7_3 = tf.nn.relu(tf.matmul(l6_3, w7[3]) + b7[3])
    l7_3 = tf.nn.dropout(l7_3, p_keep_conv)
    l7_4 = tf.nn.relu(tf.matmul(l6_4, w7[4]) + b7[4])
    l7_4 = tf.nn.dropout(l7_4, p_keep_conv)
    l7_5 = tf.nn.relu(tf.matmul(l6_5, w7[5]) + b7[5])
    l7_5 = tf.nn.dropout(l7_5, p_keep_conv)
    l7_6 = tf.nn.relu(tf.matmul(l6_6, w7[6]) + b7[6])
    l7_6 = tf.nn.dropout(l7_6, p_keep_conv)
    l7_7 = tf.nn.relu(tf.matmul(l6_7, w7[7]) + b7[7])
    l7_7 = tf.nn.dropout(l7_7, p_keep_conv)
    l7_8 = tf.nn.relu(tf.matmul(l6_8, w7[8]) + b7[8])
    l7_8 = tf.nn.dropout(l7_8, p_keep_conv)
    l7_9 = tf.nn.relu(tf.matmul(l6_9, w7[9]) + b7[9])
    l7_9 = tf.nn.dropout(l7_9, p_keep_conv)
    l7_10 = tf.nn.relu(tf.matmul(l6_10, w7[10]) + b7[10])
    l7_10 = tf.nn.dropout(l7_10, p_keep_conv)
    l7_11 = tf.nn.relu(tf.matmul(l6_11, w7[11]) + b7[11])
    l7_11 = tf.nn.dropout(l7_11, p_keep_conv)

    layer_0 = tf.matmul(l7_0, w8[0]) + b8[0]
    layer_1 = tf.matmul(l7_1, w8[1]) + b8[1]
    layer_2 = tf.matmul(l7_2, w8[2]) + b8[2]
    layer_3 = tf.matmul(l7_3, w8[3]) + b8[3]
    layer_4 = tf.matmul(l7_4, w8[4]) + b8[4]
    layer_5 = tf.matmul(l7_5, w8[5]) + b8[5]
    layer_6 = tf.matmul(l7_6, w8[6]) + b8[6]
    layer_7 = tf.matmul(l7_7, w8[7]) + b8[7]
    layer_8 = tf.matmul(l7_8, w8[8]) + b8[8]
    layer_9 = tf.matmul(l7_9, w8[9]) + b8[9]
    layer_10 = tf.matmul(l7_10, w8[10]) + b8[10]
    layer_11 = tf.matmul(l7_11, w8[11]) + b8[11]

    layer = tf.concat([layer_0, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9, layer_10, layer_11], 1)
    return tf.reshape(layer, shape=(-1, 12, 11))


def main():
    with tf.Graph().as_default() as graph:
        X = tf.placeholder("float", [None, 1, 8027, 1], name='x-input')
        Y = tf.placeholder("float", [None, 12, 11], name='y-input')
        w = init_weight([1, 25, 1, 32])
        w1 = init_weight([1, 25, 32, 64])
        w2 = init_weight([1, 3, 64, 64])
        w3 = init_weight([1, 3, 64, 64])
        w4 = init_weight([1, 3, 64, 64])
        w5 = init_weight([12, 1, 3, 64, 64])
        w6 = init_weight([12, 1, 3, 64, 64])
        w7, b7 = get_weights_bases_wout([12, 1*21*64, 100])
        w8, b8 = get_weights_bases_wout([12, 100, 11])
        p_keep_conv = tf.placeholder("float", name='p_keep_conv')
        p_keep_within = tf.placeholder("float", name='P_keep_within')
        py_x = model(X, w, w1, w2, w3, w4, w5, w6, w7, b7, w8, b8, p_keep_conv, p_keep_within)

        cost = loss_func(py_x, Y)
        train_op = tf.train.AdamOptimizer().minimize(cost)
        predict_op = py_x
        saver = tf.train.Saver()

    # with tf.Session(graph=graph) as sess:
    #     tf.global_variables_initializer().run()
    #     tf.local_variables_initializer().run()
    #     # saver.restore(sess, './model_01/model.cpkt-23')
    #     trX, trY_ = rd.get_data_max()
    #     trY = y_handle(trY_)
    #     tr_jiance = y_reduction(trY)
    #     mse_min = 0.06

    #     for i in range(500):
    #         yuce_Yr, train_op_, cost_train = sess.run(
    #             [predict_op, train_op, cost],
    #             feed_dict={X: trX, Y: trY, p_keep_conv: 0.8, p_keep_within: 0.5})

    #         yuce_Yr = sess.run(predict_op,
    #                            feed_dict={X: trX, Y: trY, p_keep_conv: 1.0,
    #                                       p_keep_within: 1.0})

    #         yc_Yr = y_reduction(yuce_Yr)
    #         mse_train = 0
    #         for j in range(500):
    #             lab = float(str(tr_jiance[j])[0] + '.' + str(tr_jiance[j])[1:13])
    #             lab_ = float(str(yc_Yr[j])[0] + '.' + str(yc_Yr[j])[1:13])
    #             mse_train = mse_train + (lab_ - lab) ** 2
    #         mse_t = mse_train/500
    #         print(i, mse_t, mse_min)
    #         if mse_t < mse_min:
    #             mse_min = mse_t
    #             saver.save(sess, "./model_01/model.cpkt", global_step=i)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, './model_01/model.cpkt-27')
        trX = get_data()

        yuce_Y = sess.run(predict_op, feed_dict={X: trX, p_keep_conv: 1.0, p_keep_within: 1.0})
        yc_Y = y_reduction(yuce_Y)
        lab = list()

        with open("./测试A-答案模板.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i in reader:
                lab.append(i)

        with open("test_A_07.csv", "w") as datacsv:
            csvwriter = csv.writer(datacsv, delattr)
            for i in range(100):
                csvwriter.writerow([lab[i][0], float(str(yc_Y[i])[0] + '.' + str(yc_Y[i])[1:13])])


if __name__ == "__main__":
    main()



