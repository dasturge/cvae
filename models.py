import tensorflow as tf
import numpy as np
keras = tf.keras
K = keras.backend


def parameters(*arams, **params):

    p = {
        'conv1_filters': 32,
        'conv2_filters': 32,
        'deconv1_filters': 32,
        'n_latent': 256,
        'layer_depth': 2,
        'kernel_size': (3, 3, 3),
        'learning_rate': 1e-3
    }
    p.update(params)

    return p


def generate_estimator(model_generator, **params):
    model = model_generator(**params)
    return keras.estimator.model_to_estimator(model)


def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def encoder(X, **params):

    # apply convolutions and maxpool layers
    ccp = X
    for i in range(params['layer_depth']):
        ccp = convconvpool(ccp, **params)

    flat = keras.layers.Flatten()(ccp)

    mean = keras.layers.Dense(
        units=params['n_latent']
    )(flat)
    var = keras.layers.Dense(
        units=params['n_latent']
    )(flat)

    z  = keras.layers.Lambda(
        sampling,
        output_shape=(params['n_latent'],),
        name='z'
    )([mean, var])

    # return conv4 for shape purposes
    return z, mean, var, ccp


def decoder(z, **params):

    fc1 = keras.layers.Dense(
        units=params['prezsize'],
        activation='relu'  # don't know if I need activation here...
    )(z)

    ucc = keras.layers.Reshape((*params['prezshape'][1:-1], 1))(fc1)

    for i in range(params['layer_depth']):
        ucc = upconvconv(ucc, **params)

    return ucc


def generate_variational_autoencoder(**params):

    X = keras.layers.Input(shape=params['input_shape'])

    z, mean, var, conv4 = encoder(X, **params)

    # get final image layer size
    tmpmodel = keras.Model(X, conv4)
    params['prezshape'] = tmpmodel.output_shape
    params['prezsize'] = np.product(params['prezshape'][1:-1])
    del tmpmodel

    y  = decoder(z, **params)

    vae = keras.Model(X, y)

    def vae_loss(x, x_decoded_mean):
        xent_loss = np.product(params['input_shape']) * keras.losses.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
        return xent_loss + kl_loss

    optimizer = keras.optimizers.Adam(lr=params['learning_rate'])

    vae.compile(optimizer=optimizer, loss=vae_loss)

    return vae


def upconvconv(input_layer, **params):

    upconv1 = keras.layers.Conv3DTranspose(
        filters=params['deconv1_filters'],
        kernel_size=params['kernel_size']
    )(input_layer)
    conv1 = keras.layers.Conv3D(
        filters=params['conv1_filters'],
        kernel_size=params['kernel_size'],
        activation='relu'
    )(upconv1)
    conv2 = keras.layers.Conv3D(
        filters=params['conv2_filters'],
        kernel_size=params['kernel_size'],
        activation='relu'
    )(conv1)

    return conv2


def convconvpool(input_layer, **params):

    conv1 = keras.layers.Conv3D(
        filters=params['conv1_filters'],
        kernel_size=params['kernel_size'],
        activation='relu'
    )(input_layer)
    conv2 = keras.layers.Conv3D(
        filters=params['conv2_filters'],
        kernel_size=params['kernel_size'],
        activation='relu'
    )(conv1)
    maxpool1 = keras.layers.MaxPool3D(strides=2)(conv2)

    return maxpool1


def generate_conditional_vae(**params):
    pass


def generate_discriminator(**params):
    pass


def plot(model, filename='model.png'):
    keras.utils.plot_model(model, to_file=filename)
