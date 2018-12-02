#!/usr/bin/env python3
import sklearn
import skopt
from skopt.space import Integer, Real
from skopt.utils import use_named_args

import inputs
import models

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


def hyperparameter_optimization(record_files, test_record):

    dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform', name='learning_rate')
    dim_layer_depth = Integer(1, 4, name='layer_depth')
    dim_n_filters = Integer(2, 128, name='n_filters')
    dim_n_filters_2 = Integer(2, 128, name='n_filters_2')
    dim_n_deconv_filters = Integer(2, 128, name='n_deconv_filters')
    dim_n_latent = Integer(2, 1028, name='n_latent')
    dim_kernel_size = Integer(2, 7, name='kernel_size')

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

        m = create_model(learning_rate=learning_rate,
                             layer_depth=layer_depth, n_filters=n_filters,
                             n_filters_2=n_filters_2,
                             n_deconv_filters=n_deconv_filters,
                             n_latent=n_latent, kernel_size=kernel_size)

        # create logging and TensorBoard

        # inputs
        # here is where I could implement cross-validation
        train = inputs.image_input_fn(filenames=record_files)
        history = m.fit(x=train, epochs=20)

