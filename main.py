import utils
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import TensorBoard
import datetime
import os


def to_grey(X):
    return X.reshape(*(X.shape + (1,)))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = to_grey(x_train)
x_test = to_grey(x_test)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

log_dir = './logs'
model_dir = './models'

os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

kernel_sizes = [3]
activations = ['relu']
input_shape = x_train.shape[1:]

conv2ds = [
    [64, 32]
]
epochs = [3]

now = datetime.datetime.now()

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

    model_filename = 'model_f%s_kernel%d_a_%s_epochs_%d_%s.model' % ('_'.join(map(str, conv2d)), kernel_size, activation, nb_epochs, now.strftime('%Y_%m_%d_%H_%M_%S'))

    model.save(os.path.join(model_dir, model_filename))
