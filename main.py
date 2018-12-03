import argparse
import os

import numpy as np

import models

def _cli():
    parser = generate_parser()
    args = parser.parse_args()

    if args.mock:
        args.data = np.random.randn(176, 256, 256, 1)
    return main(args.data)


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data')
    parser.add_argument('--mock', action='store_true')

    return parser


def main(data):

    params = {'input_shape': data.shape}
    p = models.parameters()
    p.update(params)
    m = models.generate_variational_autoencoder(**p)

    for layer in m.layers:
        print('%s %s' %(layer.output.shape, np.product(layer.output.shape[1:])))

    models.plot(m)


if __name__ == '__main__':
    _cli()
