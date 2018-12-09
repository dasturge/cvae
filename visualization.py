import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # can't use gpu on this machine for some reason.

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import models
import imloader


here = os.path.dirname(os.path.realpath(__file__))
best_model = os.path.join(here, 'model', 'best_model.keras')


def load_model(model=best_model):

    # i don't know how to retrieve my kl loss function easily.
    loss = tf.keras.losses.mean_squared_error
    model = keras.models.load_model(model, custom_objects={'loss': loss})

    return model


def plot_images(images):
    N = int(len(images) ** .5)

    plt.figure()
    for idx, img in enumerate(images):
        if idx >= N * N:
            print('not printing excess images')
            break
        plt.subplot(N, N, idx + 1)
        plt.imshow(img.T[:, ::-1])
    plt.show()


def plot_predicted_images(images, model=best_model):
    N = int(len(images) ** .5)
    if isinstance(model, str):
        model = load_model(model)

    plt.figure()
    for idx, img in enumerate(images):
        if idx >= N * N:
            print('not printing excess images')
            break

        output = get_layer_outputs(img, model, output_layer='Xout')[0]
        img2 = np.squeeze(output)
        plt.subplot(N, N, idx + 1)
        image = np.hstack((img.T[:, ::-1], img2.T[:, ::-1]))
        plt.imshow(image)

    plt.show()


def plot_2d_manifold(n=10, model=best_model, idx1=0, idx2=1):
    if isinstance(model, str):
        model = load_model(model)
    latent_dim = model.get_layer('z').output_shape[1]

    z1 = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, n))
    z2 = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, n))
    ax_grid = np.dstack(np.meshgrid(z1, z2))
    ax_grid_batch = ax_grid.reshape(n * n, 2)
    z_input = np.zeros((n * n, latent_dim))
    z_input[:, idx1] = ax_grid_batch[:, 0]
    z_input[:, idx2] = ax_grid_batch[:, 1]

    # i hope the model can handle this many images
    xout_grid = get_layer_outputs(z_input, model, 'z', 'Xout').\
        reshape(n, n, 218, 182)
    plt.figure(figsize=(n, n))
    plt.imshow(np.block(list(map(list, xout_grid))), cmap='gray')
    plt.show()


def get_layer_outputs(image, model=best_model, input_layer='X', output_layer='Xout'):
    if isinstance(model, str):
        model = keras.models.load_model(model)
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
        image = image[..., np.newaxis]
    input = model.get_layer(input_layer).input

    out = keras.Model(inputs=input, outputs=model.get_layer(output_layer).output)
    layer_output = out.predict(image)

    return layer_output


if __name__ == '__main__':
    images = imloader.load_some_images()
    model = load_model()
    plot_predicted_images(images, model)

    # plot_2d_manifold(10, model)
