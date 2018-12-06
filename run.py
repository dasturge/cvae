#!/usr/bin/env python3
import os
import traceback

import sklearn
import skopt
from skopt.space import Integer, Real
from skopt.utils import use_named_args
import tensorflow as tf

import inputs
import models

K = tf.keras.backend
TensorBoard = tf.keras.callbacks.TensorBoard
model_generator = models.generate_variational_autoencoder


def create_model(learning_rate, layer_depth, n_filters, n_filters_2,
                 n_deconv_filters, n_latent, kernel_size):
    p = {
        'conv1_filters': n_filters,
        'conv2_filters': n_filters_2,
        'deconv1_filters': n_deconv_filters,
        'n_latent': n_latent,
        'layer_depth': layer_depth,
        'kernel_size': (kernel_size, kernel_size, kernel_size),
        'learning_rate': learning_rate
    }
    params = models.parameters(**p)
    m = model_generator(**params)

    return m


def hyperparameter_optimization(record_files, test_record, working_dir='./'):

    best_accuracy = 0.0
    best_model = os.path.join(working_dir, 'best_model.keras')
    default_params = (1e-2, 3, 32, 32, 32, 256, 3)

    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_layer_depth = Integer(3, 4, name='layer_depth')
    dim_n_filters = Integer(2, 32, name='n_filters')
    dim_n_filters_2 = Integer(2, 128, name='n_filters_2')
    dim_n_deconv_filters = Integer(2, 128, name='n_deconv_filters')
    dim_n_latent = Integer(2, 512, name='n_latent')
    dim_kernel_size = Integer(2, 5, name='kernel_size')

    dimensions = [
        dim_learning_rate,
        dim_layer_depth,
        dim_n_filters,
        dim_n_filters_2,
        dim_n_deconv_filters,
        dim_n_latent,
        dim_kernel_size
    ]

    @use_named_args(dimensions=dimensions)
    def fitness(learning_rate, layer_depth, n_filters, n_filters_2,
                n_deconv_filters, n_latent, kernel_size):
        """

        :param learning_rate:
        :param layer_depth:
        :param n_filters:
        :param n_filters_2:
        :param n_deconv_filters:
        :param n_latent:
        :param kernel_size:
        :return:
        """

        m = create_model(
            learning_rate=learning_rate, layer_depth=layer_depth,
            n_filters=n_filters, n_filters_2=n_filters_2,
            n_deconv_filters=n_deconv_filters, n_latent=n_latent,
            kernel_size=kernel_size
        )

        # create logging and TensorBoard
        dirname = f'./logs/lr_{learning_rate:.0e}_layers_{layer_depth}' \
                  f'_f_{n_filters}_2f_{n_filters_2}_df_{n_deconv_filters}' \
                  f'_lat_{n_latent}_k_{kernel_size}/'
        callback_log = TensorBoard(
                log_dir=dirname, histogram_freq=0, batch_size=4,
                write_graph=True, write_grads=False, write_images=False)

        # inputs
        # here is where I could implement cross-validation
        train, train2 = inputs.image_input_fn(filenames=record_files, train=True)
        test, test2 = inputs.image_input_fn(filenames=test_record, train=False)
        try:
            history = m.fit(x=train, y=train2, epochs=5, validation_data=(test, test2),
                            steps_per_epoch=int(3008*.9/4), callbacks=[callback_log])
        except Exception as e:
            traceback.print_exc()
            return 0.0
        accuracy = history.history['val_acc'][-1]

        # Print the classification accuracy.
        print("Accuracy: {0:.2%}".format(accuracy))

        # get the previous best
        global best_accuracy

        # If the classification accuracy of the saved model is improved ...
        if accuracy > best_accuracy:
            # Save the new model to disk
            m.save(best_model)

            # Update the classification accuracy.
            best_accuracy = accuracy

        # clear data from memory
        del m
        K.clear_session()

        return -accuracy

    search_result = skopt.gp_minimize(
        func=fitness, dimensions=dimensions, acq_func='EI', n_calls=40,
        x0=default_params
    )
    print(search_result.x)
