#!/usr/bin/env python3
# standard lib
import os
import traceback

# external libs
import skopt
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # can't use gpu on this machine for some reason.
import tensorflow as tf
import numpy as np

# internal
import inputs
import models

# define some shorthands
K = tf.keras.backend
TensorBoard = tf.keras.callbacks.TensorBoard
model_generator = models.generate_variational_autoencoder
best_val_mse = None
best_val_loss = None
best_train_loss = None


def create_model(layer_depth, n_filters, n_filters_2,
                 filter_factor, n_latent):
    p = {
        'conv1_filters': n_filters,
        'conv2_filters': n_filters_2,
        'filter_factor': filter_factor,
        'n_latent': n_latent,
        'layer_depth': layer_depth,
        'kernel_size': (4, 4),
        'learning_rate': 1e-4,
        'regularizer': 'none',
        'dropout': 0,
        'input_shape': [182, 218, 1]
    }
    params = models.parameters(**p)
    m = model_generator(**params)

    return m


def hyperparameter_optimization(record_files, test_record, working_dir='./', n_jobs=1):

    global best_val_mse
    best_val_mse = 1.0
    global best_val_loss
    best_val_loss = 100000000.
    best_model = os.path.join(working_dir, 'best_model.keras')

    default_params = (7, 16, 16, 2.2, 400)  # (8e-4, 8, 20, 20, 1.4, 428, 4, 'none')

    # dim_learning_rate = Real(low=1e-5, high=1e-3, prior='log-uniform', name='learning_rate')
    dim_layer_depth = Integer(6, 8, name='layer_depth')
    dim_n_filters = Integer(2, 48, name='n_filters')
    dim_n_filters_2 = Integer(2, 48, name='n_filters_2')
    dim_filter_factor = Real(1.5, 2.25, name='filter_factor')
    dim_n_latent = Integer(400, 1024, name='n_latent')
    # dim_kernel_size = Integer(4, 5, name='kernel_size')
    # dim_regularizer = Categorical(categories=['l1', 'none'], name='regularizer')
    # dim_dropout = Integer(0, 1, name='dropout')  #  param x 0.025

    dimensions = [
        # dim_learning_rate,
        dim_layer_depth,
        dim_n_filters,
        dim_n_filters_2,
        dim_filter_factor,
        dim_n_latent,
        # dim_kernel_size,
        # dim_regularizer,
        # dim_dropout
    ]

    @use_named_args(dimensions=dimensions)
    def fitness(layer_depth, n_filters, n_filters_2,
                filter_factor, n_latent):
        """

        :param learning_rate:
        :param layer_depth:
        :param n_filters:
        :param n_filters_2:
        :param filter_factor:
        :param n_latent:
        :param kernel_size:
        :param regularizer:
        :param dropout:
        :return:
        """

        m = create_model(
            layer_depth=layer_depth,
            n_filters=n_filters, n_filters_2=n_filters_2,
            filter_factor=filter_factor, n_latent=n_latent
        )

        # create logging and TensorBoard
        batch_size = 8
        dirname = f'./logs/layers_{layer_depth}' \
                  f'_f_{n_filters}_2f_{n_filters_2}_ff_{filter_factor}' \
                  f'_lat_{n_latent}/'
        callback_log = TensorBoard(
                log_dir=dirname, histogram_freq=0, batch_size=batch_size,
                write_graph=True, write_grads=False, write_images=False)

        # inputs

        # here is where I could implement cross-validation
        train, train2 = inputs.image_input_fn(filenames=record_files, train=True, batch_size=batch_size)
        test, test2 = inputs.image_input_fn(filenames=test_record, train=False, batch_size=batch_size)
        try:
            history = m.fit(x=train, y=train2, epochs=7, validation_data=(test, test2),
                        steps_per_epoch=int(2445*.9/batch_size) + 1, callbacks=[callback_log],
                        validation_steps=int(2445*.1/batch_size) + 1)
            mse = history.history['val_mean_squared_error'][-1]
        except Exception as e:
            print('failed with params:')
            print((layer_depth, n_filters, n_filters_2,
                filter_factor, n_latent))
            traceback.print_exc()
            del m  # this fixes OOM crashes I was getting I believe
            K.clear_session()
            return 10.0  # large mse for params that can't work

        # Print the classification accuracy.
        print("MSE: {0:.2%}".format(mse))

        # get the previous best
        global best_val_mse
        global best_val_loss

        # If the classification accuracy of the saved model is improved ...
        if mse < best_val_mse:
            # Save the new model to disk
            print('saving new best model')
            m.save(best_model, overwrite=True)

            # print tensorboard dirname
            print(dirname)

            # Update the classification accuracy.
            best_val_mse = mse
        if history.history['val_loss'][-1] < best_val_loss:
            print('saving best loss in a model')
            m.save('./model/best_loss.keras')
            best_val_loss = history.history['val_loss'][-1]

        if mse > 100.0 or np.isnan(mse):
            mse = 100.0  # trim down massive overfits

        # clear data from memory
        del m
        K.clear_session()

        return mse

    search_result = skopt.gp_minimize(
        func=fitness, dimensions=dimensions, acq_func='EI', n_calls=50,
        x0=default_params, n_jobs=n_jobs
    )
    print(search_result.x)
