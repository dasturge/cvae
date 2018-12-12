import tensorflow as tf
import numpy as np
keras = tf.keras
K = keras.backend


def parameters(*arams, **params):

    p = {
        'conv1_filters': 32,
        'conv2_filters': 32,
        'deconv1_filters': 32,
        'filter_factor': 1.5,
        'n_latent': 256,
        'layer_depth': 3,
        'kernel_size': (3, 3),
        'learning_rate': 1e-3,
        'input_shape': [256, 256, 1]
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
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def encoder(X, **params):

    # apply convolutions and maxpool layers
    ccp = X
    odd_dims = []
    for i in range(params['layer_depth']):
        odd_dims.append([x for x in ccp.shape[1:-1]])
        ccp = convconvpool(ccp, ffactor=params['filter_factor'] ** i, **params)

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
    return z, mean, var, ccp, odd_dims


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
        ffactor = params['filter_factor'] ** (params['layer_depth'] - i - 1)
        dim = params['layer_dims'][params['layer_depth'] - i - 1]
        ucc = upconvconv(ucc, dim=dim, ffactor=ffactor, **params)

    #out = keras.layers.Conv2D(
    #    filters=params['conv1_filters'],
    #    kernel_size=params['kernel_size'],
    #    padding='same',
    #    activation='relu'
    #)(ucc)
    out = keras.layers.Conv2D(
        filters=1,
        kernel_size=params['kernel_size'],
        padding='same',
        name='Xout',
    )(ucc)

    return out


def generate_variational_autoencoder(**params):

    X = keras.layers.Input(shape=params['input_shape'], name='X')
    
    z, mean, var, conv4, layer_dims = encoder(X, **params)

    # get final image layer size
    tmpmodel = keras.Model(X, conv4)
    params['prezshape'] = tmpmodel.output_shape
    params['prezsize'] = np.product(params['prezshape'][1:-1])
    del tmpmodel

    y = decoder(z, layer_dims=layer_dims, **params)

    vae = keras.Model(X, y)

    def loss(x, x_decoded):
        sse = K.sum(K.square(x - x_decoded), axis=[1, 2, 3])
        dx = tf.image.image_gradients(x)
        dx_decoded = tf.image.image_gradients(x_decoded)
        ssed = K.sum(K.square(dx[0] - dx_decoded[0]) + K.square(dx[1] - dx_decoded[1]), axis=[1, 2, 3])
        # cc = local_cc(x, x_decoded_mean)
        # mi = tf.py_func(mutual_information, [x, x_decoded_mean], Tout=[tf.float32])
        kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)

        return sse + ssed * .8 + 0.1 * kl_loss

    def loss_cc(x, x_decoded):
        sse = K.sum(K.square(x - x_decoded), axis=[1, 2, 3])
        cc = local_cc(x, x_decoded, kernel_size=3)
        kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
        return sse + kl_loss - cc * .0001

    def loss_mi(x, x_decoded):
        mi = mutual_information(x, x_decoded)
        kl_loss = - 0.5 * K.sum(1 + var - K.square(mean) - K.exp(var), axis=-1)
        return kl_loss - mi

    optimizer = keras.optimizers.Adam(lr=params['learning_rate'])

    vae.compile(optimizer=optimizer, loss=loss_cc, metrics=['mse'])

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

    odds = [x % 2 for x in params['dim']]
    dim = params['dim']

        # kwargs['output_padding'] = [int(o) for o in odds]

    upconv1 = keras.layers.Conv2DTranspose(
        filters=int(params['conv1_filters'] * params['ffactor']),
        kernel_size=params['kernel_size'],
        strides=2,
        padding='same',
        **kwargs
    )(input_layer)

    if sum(odds):
        upconv1 = keras.layers.Lambda(
            lambda x: x[:, :dim[0], :dim[1], :],
            output_shape=(upconv1.shape[0], *dim, upconv1.shape[-1])
        )(upconv1)

    #conv1 = keras.layers.Conv2D(
    #    filters=params['conv1_filters'],
    #    kernel_size=params['kernel_size'],
    #    padding='same',
    #    **kwargs
    #)(upconv1)
    bn2 = keras.layers.BatchNormalization()(upconv1)
    lrelu1 = keras.layers.LeakyReLU(alpha=0.1)(bn2)
    if params['dropout']:
        lrelu1 = keras.layers.Dropout(rate=params['dropout'])(lrelu1)
    conv2 = keras.layers.Conv2D(
        filters=int(params['conv2_filters'] * params['ffactor']),
        kernel_size=params['kernel_size'],
        padding='same',
        **kwargs
    )(lrelu1)
    bn3 = keras.layers.BatchNormalization()(conv2)
    lrelu2 = keras.layers.LeakyReLU(alpha=0.1)(bn3)
    if params['dropout']:
        lrelu2 = keras.layers.Dropout(rate=params['dropout'])(lrelu2)

    return lrelu2


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
        filters=int(params['conv2_filters'] * params['ffactor']),
        kernel_size=params['kernel_size'],
        padding='same',
        **kwargs
    )(input_layer)
    bn1 = keras.layers.BatchNormalization()(conv1)
    lrelu1 = keras.layers.LeakyReLU(alpha=0.1)(bn1)
    if params['dropout']:
        lrelu1 = keras.layers.Dropout(rate=params['dropout'])(lrelu1)
    conv2 = keras.layers.Conv2D(
        filters=int(params['conv1_filters'] * params['ffactor']),
        kernel_size=params['kernel_size'],
        padding='same',
        strides=(2, 2),
        **kwargs
    )(lrelu1)
    bn2 = keras.layers.BatchNormalization()(conv2)
    lrelu2 = keras.layers.LeakyReLU(alpha=0.1)(bn2)
    if params['dropout']:
        lrelu2 = keras.layers.Dropout(rate=params['dropout'])(lrelu2)

    return lrelu2


def extract_decoder(model, name='dense_2'):
    """
    extracts decoder, only works in sequential case of functional api
    :param model: loaded keras model
    :param name: new input layer
    :return: model with same output but new input layer.
    """
    not_found = True
    for layer in model.layers:
        if not_found and layer.name != name:
            continue
        elif not_found:
            not_found = False
            input_layer = keras.Input(shape=layer.input.shape[1:],
                                      dtype=layer.input.dtype)
            current_layer = input_layer
        current_layer = layer(current_layer)
    new_model = keras.Model(inputs=input_layer, outputs=current_layer)
    return new_model


def generate_conditional_vae(**params):
    pass


def generate_discriminator(**params):
    pass


def plot(model, filename='model.png'):
    keras.utils.plot_model(model, to_file=filename)


def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1, 1])
  return tf.constant(a, dtype=1)


def laplace(x):
  """Compute the 2D laplacian of an array"""
  laplace_filter = make_kernel([[0.5, 1.0, 0.5],
                                [1.0, -6., 1.0],
                                [0.5, 1.0, 0.5]])
  return tf.nn.conv2d(x, laplace_filter, strides=[1, 1, 1, 1], padding='SAME')


def local_cc(x, y, kernel_size=3, padding='VALID'):
    """
    compute the local cross correlation of two images
    :param x: im1
    :param y: im2
    :param kernel_size: kernel size for sliding cross correlation
    :param padding: one of 'VALID' or 'SAME', case sensitive.
    :return: cross_correlation image
    """
    # get kernel patches
    x_patches = tf.extract_image_patches(
        x, [1, kernel_size, kernel_size, 1], [1, 1, 1, 1], [1, 1, 1, 1], 
        padding=padding)
    y_patches = tf.extract_image_patches(
        y, [1, kernel_size, kernel_size, 1], [1, 1, 1, 1], [1, 1, 1, 1],
        padding=padding)

    # normalize patches
    mean, var = tf.nn.moments(x_patches, axes=[1, 2], keep_dims=True)
    x_patches = (x_patches - mean) / var
    mean, var = tf.nn.moments(y_patches, axes=[1, 2], keep_dims=True)
    y_patches = (y_patches - mean) / var

    # maps a 2d convolution (actually cross-correlation) over batches of image patches for x and y.
    lcc = tf.map_fn(
        image_cc,
        elems=[x_patches, y_patches],
        dtype=tf.float32
    )

    # sum cross correlation and average over kernel size
    total_cc = tf.reduce_sum(lcc) / kernel_size ** 2

    return total_cc


def image_cc(z):
    return tf.nn.conv2d(
            tf.expand_dims(z[0], 0),
            tf.expand_dims(z[1], 3),
            strides=[1, 1, 1, 1],
            padding="VALID")


def mutual_information(a, b):
    mi = K.map_fn(
        lambda x: keras.layers.Lambda(_mutual_information)(x),
        elems=[a, b],
        dtype=tf.float32
    )
    mi = K.sum(mi)
    return mi


def _mutual_information(a, b):
    # compute joint histogram
    histogram, *_ = np.histogram2d(a.flatten(), b.flatten(), bins=20)
    # joint pdf
    pxy = histogram / float(np.sum(histogram))
    # compute partials
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nonzero = pxy[pxy > 0]
    mi = np.sum(nonzero * np.log(nonzero / px_py[pxy > 0]))
    return mi.astype(np.float32)
