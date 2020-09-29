import numpy as np
import tensorflow as tf
import scipy.io as sio
from glob import glob
import os
from monty.collections import AttrDict


def load_data_from_mat(path):
    data = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    for key in data:
        if isinstance(data[key], sio.matlab.mio5_params.mat_struct):
            data[key] = _todict(data[key])
    return data


def _todict(matobj):
    # A recursive function which constructs from matobjects nested dictionaries
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def affnist_reader(batch_size):
    test_path = glob(os.path.join('./data/affnist/', "test.mat"))
    print(test_path)
    test_data = load_data_from_mat(test_path[0])
    testX = test_data['affNISTdata']['image'].transpose()
    testY = test_data['affNISTdata']['label_int']

    testX = testX.reshape((320000, 40, 40, 1)).astype(np.float32)
    testY = testY.reshape((320000)).astype(np.int32)

    X = tf.convert_to_tensor(testX, dtype=tf.float32) / 255.
    Y = tf.convert_to_tensor(testY, dtype=tf.int64)

    input_queue = tf.train.slice_input_producer([X, Y], shuffle=True)
    images = tf.image.resize_images(input_queue[0], [40, 40])
    labels = input_queue[1]
    X, Y = tf.train.batch([images, labels], batch_size=batch_size)

    testset = AttrDict(image=X, label=Y)
    return testset
