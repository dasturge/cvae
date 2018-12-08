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
        'layer_depth': 3,
        'kernel_size': (3, 3),
        'learning_rate': 1e-3,
        'input_shape': [192, 224, 1]
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

    z = keras.layers.Lambda(
        sampling,
        output_shape=(params['n_latent'],),
        name='z'
    )([mean, var])

    # return conv4 for shape purposes
    return z, mean, var, ccp


def decoder(z, **params):

    kwargs = {}
    rfactor = 1e-2
    if params.get('regularizer') == 'l2':
        kwargs['activity_regularizer'] = keras.regularizers.l2(rfactor)
    elif params.get('regularizer') == 'l1':
        kwargs['activity_regularizer'] = keras.regularizers.l1(rfactor)
    elif params.get('regularizer') == 'both':
        kwargs['activity_regularizer'] = keras.regularizers.l1_l2(l1=rfactor, l2=rfactor)

    fc1 = keras.layers.Dense(
        units=params['prezsize'],
        activation='relu',  # don't know if I need activation here...
        **kwargs
    )(z)

    ucc = keras.layers.Reshape((*params['prezshape'][1:-1], 1))(fc1)

    for i in range(params['layer_depth']):
        ucc = upconvconv(ucc, **params)

    out = keras.layers.Conv2D(
        filters=1,
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        name='Xout'
    )(ucc)

    return out


def generate_variational_autoencoder(**params):

    X = keras.layers.Input(shape=params['input_shape'], name='X')
    
    z, mean, var, conv4 = encoder(X, **params)

    # get final image layer size
    tmpmodel = keras.Model(X, conv4)
    params['prezshape'] = tmpmodel.output_shape
    params['prezsize'] = np.product(params['prezshape'][1:-1])
    del tmpmodel

    y = decoder(z, **params)

    vae = keras.Model(X, y)

    def loss(x, x_decoded_mean):
        rec_loss = np.product(params['input_shape']) * keras.losses.\
                mean_squared_error(x, x_decoded_mean)
        rec_loss = K.mean(rec_loss)
        kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
        return rec_loss + kl_loss

    optimizer = keras.optimizers.Adam(lr=params['learning_rate'])

    vae.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    return vae


def upconvconv(input_layer, **params):

    kwargs = {}
    rfactor = 1e-2
    if params.get('regularizer') == 'l2':
        kwargs['activity_regularizer'] = keras.regularizers.l2(rfactor)
    elif params.get('regularizer') == 'l1':
        kwargs['activity_regularizer'] = keras.regularizers.l1(rfactor)
    elif params.get('regularizer') == 'both':
        kwargs['activity_regularizer'] = keras.regularizers.l1_l2(l1=rfactor, l2=rfactor)

    upconv1 = keras.layers.Conv2DTranspose(
        filters=params['deconv1_filters'],
        kernel_size=params['kernel_size'],
        strides=2,
        padding='same',
        **kwargs
    )(input_layer)
    conv1 = keras.layers.Conv2D(
        filters=params['conv1_filters'],
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        **kwargs
    )(upconv1)
    conv2 = keras.layers.Conv2D(
        filters=params['conv2_filters'],
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        **kwargs
    )(conv1)

    return conv2


def convconvpool(input_layer, **params):

    kwargs = {}
    rfactor = 1e-2
    if params.get('regularizer') == 'l2':
        kwargs['activity_regularizer'] = keras.regularizers.l2(rfactor)
    elif params.get('regularizer') == 'l1':
        kwargs['activity_regularizer'] = keras.regularizers.l1(rfactor)
    elif params.get('regularizer') == 'both':
        kwargs['activity_regularizer'] = keras.regularizers.l1_l2(l1=rfactor, l2=rfactor)

    conv1 = keras.layers.Conv2D(
        filters=params['conv1_filters'],
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        **kwargs
    )(input_layer)
    conv2 = keras.layers.Conv2D(
        filters=params['conv2_filters'],
        kernel_size=params['kernel_size'],
        activation='relu',
        padding='same',
        **kwargs
    )(conv1)
    maxpool1 = keras.layers.MaxPool2D(strides=2)(conv2)

    return maxpool1


def generate_conditional_vae(**params):
    pass


def generate_discriminator(**params):
    pass


def plot(model, filename='model.png'):
    keras.utils.plot_model(model, to_file=filename)
