from model import InceptionNet
import convert_tfrecord as ct
import tensorflow as tf
import util.train_tools as uttt


if __name__ == '__main__':
    img, lab = ct.get_data()
    train_img = tf.constant(img[0:5642].reshape((-1, 1, 40, 1)), dtype=tf.float32)
    train_lab = tf.constant(lab.reshape((-1, 1)), dtype=tf.float32)
    print(train_img, train_lab)
    vali_img = tf.constant(img[5642:].reshape((-1, 1, 40, 1)), dtype=tf.float32)
    print(vali_img)
    uttt.train(InceptionNet, train_img, train_lab, vali_img,
               [-1, 1, 40, 1], [-1, 1], './logs', 5, 500, 180, 180)
