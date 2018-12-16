import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # can't use gpu on this machine for some reason.

import tensorflow as tf
import tensorflow.keras as keras
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


def plot_2d_manifold(n=7, model=best_model, idx1=0, idx1b=1, idx2=1, idx2b=2):
    if isinstance(model, str):
        model = load_model(model)
    latent_dim = model.get_layer('z').output_shape[1]

    # take a normally spaced variable about 0.
    z1 = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, n))
    z2 = scipy.stats.norm.ppf(np.linspace(0.01, 0.99, n))
    # create a meshgrid of normed values to insert into latent space vector
    ax_grid = np.dstack(np.meshgrid(z1, z2))
    ax_grid_batch = ax_grid.reshape(n * n, 2)
    # create latent space inputs, NxN batch-size x latent space size
    z_input = np.zeros((n * n, latent_dim))
    # create vectors in the latent space instead of a univariate variation.
    z_input[:, idx1:idx1b] = np.repeat(ax_grid_batch[:, 0][..., np.newaxis], idx1b-idx1, axis=1)
    z_input[:, idx2:idx2b] = np.repeat(ax_grid_batch[:, 1][..., np.newaxis], idx2b-idx2, axis=1)

    # i hope the model can handle this many images
    xout_grid = get_layer_outputs(
        z_input,
        model,
        input_layer='dense_2',
        output_layer='Xout'
    ).reshape(n, n, 182, 218)
    plt.figure(figsize=(n, n))
    plt.imshow(np.block(list(map(list, xout_grid))).T, cmap='gray')
    plt.show()


def get_layer_outputs(image, model=best_model, input_layer='X', output_layer='Xout'):
    if isinstance(model, str):
        model = keras.models.load_model(model)
    if input_layer == 'X' and len(image.shape) == 2:
        # pad axes if this is supposed to be a "batch" of images
        image = image[np.newaxis, ...]
        image = image[..., np.newaxis]
    input = model.get_layer(input_layer).input
    if input_layer == 'X':
        new_model = tf.keras.Model(inputs=input, outputs=model.get_layer(output_layer).output)
    elif output_layer == 'Xout':
        # its a more complex process to extract outputs of an intermediate layer in keras
        new_model = models.extract_decoder(model, input_layer)
    else:
        print('please specify a different input_layer or output_layer')
    layer_output = new_model.predict(image)

    return layer_output


if __name__ == '__main__':
    images = imloader.load_some_images()
    model = load_model()
    # plot before/after images
    plot_predicted_images(images, model)
    # plot latent space manifold
    plot_2d_manifold(5, model, idx1=0, idx1b=20, idx2=300, idx2b=320)
