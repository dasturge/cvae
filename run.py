#!/usr/bin/env python3
# standard lib
import os
import traceback

# external libs
import skopt
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # can't use gpu on this machine for some reason.
import tensorflow as tf

# internal
import inputs
import models

# define some shorthands
K = tf.keras.backend
TensorBoard = tf.keras.callbacks.TensorBoard
model_generator = models.generate_variational_autoencoder
best_mse = None


def create_model(learning_rate, layer_depth, n_filters, n_filters_2,
                 filter_factor, n_latent, kernel_size, regularizer):
    p = {
        'conv1_filters': n_filters,
        'conv2_filters': n_filters_2,
        'filter_factor': filter_factor,
        'n_latent': n_latent,
        'layer_depth': layer_depth,
        'kernel_size': (kernel_size, kernel_size),
        'learning_rate': learning_rate,
        'regularizer': regularizer
    }
    params = models.parameters(**p)
    m = model_generator(**params)

    return m


def hyperparameter_optimization(record_files, test_record, working_dir='./', n_jobs=1):

    global best_mse
    best_mse = 1.0
    best_model = os.path.join(working_dir, 'best_model.keras')

    default_params = (1e-3, 5, 16, 16, 1.5, 100, 3, 'l2')

    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_layer_depth = Integer(2, 8, name='layer_depth')
    dim_n_filters = Integer(2, 32, name='n_filters')
    dim_n_filters_2 = Integer(2, 32, name='n_filters_2')
    dim_filter_factor = Real(1.0, 2.0, name='filter_factor')
    dim_n_latent = Integer(2, 512, name='n_latent')
    dim_kernel_size = Integer(2, 5, name='kernel_size')
    dim_regularizer = Categorical(categories=['l2', 'l1', 'none', 'both'], name='regularizer')

    dimensions = [
        dim_learning_rate,
        dim_layer_depth,
        dim_n_filters,
        dim_n_filters_2,
        dim_filter_factor,
        dim_n_latent,
        dim_kernel_size,
        dim_regularizer
    ]

    @use_named_args(dimensions=dimensions)
    def fitness(learning_rate, layer_depth, n_filters, n_filters_2,
                filter_factor, n_latent, kernel_size, regularizer):
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
            filter_factor=filter_factor, n_latent=n_latent,
            kernel_size=kernel_size, regularizer=regularizer
        )

        # create logging and TensorBoard
        batch_size = 8
        dirname = f'./logs/lr_{learning_rate:.0e}_layers_{layer_depth}' \
                  f'_f_{n_filters}_2f_{n_filters_2}_ff_{filter_factor}' \
                  f'_lat_{n_latent}_k_{kernel_size}_reg_{regularizer}/'
        callback_log = TensorBoard(
                log_dir=dirname, histogram_freq=0, batch_size=batch_size,
                write_graph=True, write_grads=False, write_images=False)

        # inputs

        # here is where I could implement cross-validation
        train, train2 = inputs.image_input_fn(filenames=record_files, train=True, batch_size=batch_size)
        test, test2 = inputs.image_input_fn(filenames=test_record, train=False, batch_size=batch_size)
        try:
            history = m.fit(x=train, y=train2, epochs=3, validation_data=(test, test2),
                        steps_per_epoch=int(2445*.9/batch_size), callbacks=[callback_log],
                        validation_steps=int(2445*.1/batch_size))
            mse = history.history['val_mean_squared_error'][-1]
        except Exception as e:
            print('failed with params:')
            print((learning_rate, layer_depth, n_filters, n_filters_2,
                filter_factor, n_latent, kernel_size, regularizer))
            traceback.print_exc()
            return 0.0

        # Print the classification accuracy.
        print("MSE: {0:.2%}".format(mse))

        # get the previous best
        global best_mse

        # If the classification accuracy of the saved model is improved ...
        if mse < best_mse:
            # Save the new model to disk
            print('saving new best model')
            m.save(best_model, overwrite=True)

            # Update the classification accuracy.
            best_mse = mse

        # clear data from memory
        del m
        K.clear_session()

        return mse

    search_result = skopt.gp_minimize(
        func=fitness, dimensions=dimensions, acq_func='EI', n_calls=50,
        x0=default_params, n_jobs=n_jobs
    )
    print(search_result.x)
