import utils
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.callbacks import TensorBoard
import numpy.random
import datetime
import os
import argparse


if __name__ == '__main__':

    # Setup the argument parser.
    parser = argparse.ArgumentParser()

    parser.add_argument('--base-path', default='./')
    parser.add_argument('--testing', default=False, action='store_true')
    parser.add_argument('--seed', default=None, type=int)

    # All possible arguments used for training, with default values.
    parser.add_argument('--kernel-sizes', default='3')
    parser.add_argument('--activations', default='relu')
    parser.add_argument('--conv2ds', default='64-32')
    parser.add_argument('--epochs', default='10')
    parser.add_argument('--dropouts', default='0')

    args = parser.parse_args()

    now = datetime.datetime.now()
    date_path = now.strftime('%Y_%m_%d_%H_%M')

    # Base path used when we saves the logs and models.
    base_path = args.base_path

    log_dir = os.path.join(base_path, 'logs', date_path)
    model_dir = os.path.join(base_path, 'models', date_path)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # If we are running in test mode, only use 500 examples.
    if args.testing:
        x_train = x_train[:500]
        x_test = x_test[:500]
        y_train = y_train[:500]
        y_test = y_test[:500]

    # Normalize the input data.
    x_train = utils.to_grey(x_train)
    x_test = utils.to_grey(x_test)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    input_shape = x_train.shape[1:]

    if args.seed:
        numpy.random.seed(args.seed)


    # Setup different iterable variables from the command line that are used when trying many different combinations.
    kernel_sizes = utils.cmd_arg_to_list(args.kernel_sizes, int)
    activations = utils.cmd_arg_to_list(args.activations)

    conv2ds = utils.cmd_arg_to_list(args.conv2ds)
    conv2ds = map(lambda x: x.split('-'), conv2ds)
    conv2ds = list(conv2ds)

    for i in range(len(conv2ds)):
        conv2ds[i] = list(map(int, conv2ds[i]))

    epochs = utils.cmd_arg_to_list(args.epochs, int)
    dropouts = utils.cmd_arg_to_list(args.dropouts, float)

    # Iterate over all combinations from the variables above.
    for conv2d, kernel_size, activation, nb_epochs, dropout in utils.iter_combinations(conv2ds, kernel_sizes, activations, epochs, dropouts):

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

            if dropout:
                model.add(Dropout(dropout))

        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            TensorBoard(log_dir)
        ]

        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=nb_epochs, callbacks=callbacks)

        model_filename = 'model_f%s_kernel%d_a_%s_epochs_%d_dpo_%s.model' % ('_'.join(map(str, conv2d)), kernel_size, activation, nb_epochs, str(dropout).replace('.', '_'))

        model.save(os.path.join(model_dir, model_filename))
