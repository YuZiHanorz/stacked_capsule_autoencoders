import tensorflow as tf
from tensorflow import keras
import numpy as np
from stacked_capsule_autoencoders.capsules.data import preprocess
import time


def prepare_data_for_resnet50(data_to_transform):
    data = data_to_transform.astype('float32')
    data /= 255
    t = tf.convert_to_tensor(data, tf.float32, name='t')
    t = tf.expand_dims(t, -1)
    data = preprocess.pad_and_shift(t, 40)
    data = tf.squeeze(data, -1)
    session = tf.Session()
    data = session.run(data)
    data = np.stack([data, data, data], axis=-1)
    print(data.shape)
    return data


(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.cifar10.load_data()
#x_test = prepare_data_for_resnet50(xtest)
x_test = xtest
y_test = keras.utils.to_categorical(ytest, 10)
#x_train = prepare_data_for_resnet50(xtrain)
x_train = xtrain
y_train = keras.utils.to_categorical(ytrain, 10)
print(x_test.shape, x_train.shape)

input_shape = (32, 32, 3)
classes = 10
bs = 32
inputs = keras.Input(shape=input_shape)
outputs = keras.applications.ResNet50(input_shape=input_shape, weights=None, classes=classes)(inputs)
model = keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('timing start')
st = time.time()
score1 = model.evaluate(x_test, y_test, batch_size=bs, verbose=0)
score2 = model.evaluate(x_train, y_train, batch_size=bs, verbose=0)
print('timing end')
ed = time.time()
print(bs, score1, score2)
print(ed - st)

