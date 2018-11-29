import utils
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard
import os


def to_grey(X):
    return X.reshape(*(X.shape + (1,)))

(x_train, y_train), (x_test, y_test) = mnist.load_data('./mnist.npz')

x_train = x_train[:500]
y_train = y_train[:500]

x_train = to_grey(x_train)
x_test = to_grey(x_test)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

log_dir = './logs'

os.makedirs(log_dir, exist_ok=True)

kernel_sizes = [3]
activations = ['relu']
input_shape = x_train.shape[1:]

conv2ds = [
    [64, 32]
]
epochs = [3]
for conv2d, kernel_size, activation, nb_epochs in utils.iter_combinations(conv2ds, kernel_sizes, activations, epochs):

    model = Sequential()

    model_input_shape = input_shape

    for filters in conv2d:
        conv2d_kwargs = {
            'kernel_size': kernel_size,
            'activation': activation,
            'filters': filters
        }

        if model_input_shape:
            conv2d_kwargs['input_shape'] = model_input_shape
            model_input_shape = None

        model.add(Conv2D(**conv2d_kwargs))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        TensorBoard(log_dir)
    ]

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, callbacks=callbacks)
