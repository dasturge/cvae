import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # can't use gpu on this machine for some reason.

from itertools import product


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

# import imloader


here = os.path.dirname(os.path.realpath(__file__))
best_model = os.path.join(here, 'model', 'best_model.keras')



def load_model(model=best_model):
    # i don't know how to retrieve my kl loss function easily.
    loss = tf.keras.losses.mean_squared_error
    model = keras.models.load_model(model, custom_objects={'loss': loss})

    return model


def plot_images(images, model=best_model):
    N = int(len(images) ** .5)
    # assert (N == int(N)), 'number of images should be square'

    #if isinstance(model, str):lr_3e-04_layers_3_f_36_2f_6_df_114_lat_362_k_4
    #    model = keras.models.load_model(model)

    plt.figure()
    for idx, img in enumerate(images):
        if idx >= N * N:
            print('not printing excess images')
            break
        plt.subplot(N, N, idx + 1)
        plt.imshow(img.T)
    plt.show()


def get_layer_outputs(image, model=best_model):
    if isinstance(model, str):
        model = keras.models.load_model(model)
    input = model.input
    outputs = [layer.output for layer in model.layers]
    funcs = [K.function([input], [out]) for out in outputs]

    layer_outputs = [func([image]) for func in funcs]

    return layer_outputs


if __name__ == '__main__':
    model = load_model()
    x = 1
