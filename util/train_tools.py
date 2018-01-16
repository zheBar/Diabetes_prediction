import tensorflow as tf
from tensorflow.contrib.slim.python.slim.learning import train_step
import tensorflow.contrib.slim as slim
import csv


def trainloss(o_label, l_label):
    """
    模型训练的损失（可以根据自己的需求重载该函数）
    Args:
        o_label:计算后的标签列表
        l_label:原始的标签列表

    Returns:
        total_loss:总的损失大小
    """
    with tf.name_scope('losses'):
        total_loss = tf.reduce_mean(tf.square(o_label-l_label))
        tf.summary.scalar('total_loss', total_loss)
    return total_loss


def train(model, images, labels, images_validation, x_shape, y_shape, saved_path, num_per_valid,
          numer_of_step, save_summaries_secs, save_interval_secs, trainloss_fn=trainloss):
    """

    Args:
        model: 使用的模型
        images: 训练集的图片张量
        labels: 训练集的图片标签张量
        images_validation: 验证集的图片张量
        x_shape: 需要将x转换成的shape eg.[-1, 50, 50, 3]
        y_shape: 需要将y转换成的shape eg.[-1, 52]
        saved_path: 模型保存的位置
        num_per_valid: 多少代输出一次验证集的准确率
        numer_of_step: 训练多少代
        save_summaries_secs: 多就保存一次摘要（单位为s）
        save_interval_secs: 多就保存一次模型(单位为s)
        trainloss_fn: 训练损失

    Returns:

    """
    image = tf.reshape(images, x_shape)
    label = tf.reshape(labels, y_shape)
    image_validation = tf.reshape(images_validation, x_shape)

    with tf.variable_scope("model") as scope:
        inceptionNet = model()
        predictions = inceptionNet.get_model(image)
        scope.reuse_variables()
        predictions_validation = inceptionNet.get_model(image_validation)

    accuracy_validation = predictions_validation

    def train_step_fn(session, *args, **kwargs):
        total_loss, should_stop = train_step(session, *args, **kwargs)

        if session.run(train_step_fn.step) % num_per_valid == 0:
            accuracy = session.run(train_step_fn.accuracy_validation)
            step = session.run(train_step_fn.step)
            with open("./test/test_" + str(step) + ".csv", "w") as datacsv:
                csvwriter = csv.writer(datacsv, delattr)
                for i in range(len(accuracy)):
                    csvwriter.writerow(accuracy[i])

        return [total_loss, should_stop]

    train_step_fn.step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    train_step_fn.accuracy_validation = accuracy_validation

    optimizer = tf.train.AdamOptimizer()
    total_loss = trainloss_fn(predictions, label)
    train_tensor = slim.learning.create_train_op(total_loss, optimizer, global_step=train_step_fn.step)

    slim.learning.train(
        train_tensor,
        saved_path,
        train_step_fn=train_step_fn,
        number_of_steps=numer_of_step,
        save_summaries_secs=save_summaries_secs,
        save_interval_secs=save_interval_secs)


if __name__ == '__main__':
    from recognition.en_char.model import Inception_ResNet
    from recognition.en_char.read_dataset import test_batch, train_batch

    images, labels = tf.train.batch(train_batch(), batch_size=50, dynamic_pad=True)
    images_validation, labels_validation = tf.train.batch(test_batch(), batch_size=100, dynamic_pad=True)
    train(Inception_ResNet, images, labels, images_validation,
          [-1, 50, 50, 3], [-1, 52], './logs', 1, 1000, 60, 120)
