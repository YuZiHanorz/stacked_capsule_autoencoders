import tensorflow as tf
from tensorflow import keras
import numpy as np
from stacked_capsule_autoencoders.capsules.data import preprocess


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
    return data


(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.mnist.load_data()
x_test = prepare_data_for_resnet50(xtest)
y_test = keras.utils.to_categorical(ytest, 10)
x_train = prepare_data_for_resnet50(xtrain)
y_train = keras.utils.to_categorical(ytrain, 10)

input_shape = (40, 40, 3)
classes = 10
inputs = keras.Input(shape=input_shape)
outputs = keras.applications.ResNet50(input_shape=input_shape, weights=None, classes=classes)(inputs)
model = keras.Model(inputs, outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(x_test, y_test, verbose=0)
score = model.evaluate(x_train, y_train, verbose=0)
print(score)
