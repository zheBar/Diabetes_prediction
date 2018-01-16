import csv
import numpy as np

from util.tfrecord_tools import TFRecord


train_path = './data/d_.csv'
fill = {'男': 1.0, '女': 2.0, '': 0.0, '??': 0.0}
cols_number = 40
train_num = 5078
validation_num = 564
test_num = 1000


def read_data():
    with open(train_path, "r", encoding="GB18030") as csvfile:
        read = csv.reader(csvfile)
        read = list(read)
        read = np.array(read)
        for i in range(1, len(read)):
            for j in range(1, len(read[i])):
                if read[i][j] in fill.keys():
                    read[i][j] = fill[read[i][j]]
            if i == 1:
                data = np.reshape(read[i][1:-1], (1, -1))
                label = np.reshape(read[i][-1], (-1, 1))
            elif i <= 5642:
                data = np.concatenate((data, np.reshape(read[i][1:-1], (1, -1))), axis=0)
                label = np.concatenate((label, np.reshape(read[i][-1], (-1, 1))), axis=0)
            elif i > 5642:
                data = np.concatenate((data, np.reshape(read[i][1:], (1, -1))), axis=0)
        return data, label


def min_mine(data):
    min_ = data[0]
    for i in data:
        if i != 0.0:
            min_ = min(min_, i)
    return min_


def data_prep(data, label):
    data[:, 2] = str(0.0)
    data = data.astype(float)
    label = label.astype(float)
    cols_number = len(data[0])

    for j in range(cols_number):
        max_ = max(data[:, j])
        min_ = min_mine(data[:, j])
        if (max_ - min_) == 0.0:
            data[:, j] = float(0.0)
        else:
            data[:, j] = (data[:, j] - min_) / (max_ - min_)
    train_data = np.reshape(data, [-1, 40])
    train_label = np.reshape(label, [-1, 1])
    return train_data, train_label


def to_tfrecord(data, label):
    train_tfrecords = TFRecord({'data': [float], 'label': [float]})
    writer_train = train_tfrecords.writer(
        save_dir='./data/tfrecord/train/',
        prefix='train')

    validation_tfrecords = TFRecord({'data': [float], 'label': [float]})
    writer_validation = validation_tfrecords.writer(
        save_dir='./data/tfrecord/validation/',
        prefix='validation')

    test_tfrecords = TFRecord({'data': [float], 'label': [float]})
    writer_test = test_tfrecords.writer(
        save_dir='./data/tfrecord/test/',
        prefix='test')
    sum_train = 0
    sum_validation = 0
    sum_test = 0
    sum = 0
    for i in range(5642):
        sum = sum % 10
        if sum < 9:
            writer_train.add_example({'data': data[i], 'label': label[i]})
            sum_train += 1
        elif sum == 9:
            writer_validation.add_example({'data': data[i], 'label': label[i]})
            sum_validation += 1
        sum += 1
    for i in range(5642, len(data)):
        writer_train.add_example({'data': data[i]})
        sum_test += 1
    writer_train.close()
    writer_validation.close()
    writer_test.close()
    print("sum_train = %d" % sum_train)
    print("sum_validation = %d" % sum_validation)
    print("sum_test = %d" % sum_test)
    return


def get_data():
    data, label = read_data()
    data1, label1 = data_prep(data, label)
    return data1, label1


if __name__ == "__main__":
    data, label = read_data()
    data1, label1 = data_prep(data, label)
    print(data1)
    print(label1)