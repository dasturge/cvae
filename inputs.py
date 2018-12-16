#!/usr/bin/env python3
import argparse
import os
import re

import nibabel as nb
import numpy as np
import tensorflow as tf

scanner_name = re.compile('.*/ABCD_(.*)_SEFMNoT2/.*')


def _cli():
    parser = generate_parser()
    args = parser.parse_args()

    niftis = args.niftis
    if args.list:
        assert len(niftis) == 0, 'must provide filename through text file ' \
                                 'or as space separated list. Not both.'
        with open(args.list) as fd:
            niftis = list(fd.readlines())
    else:
        assert niftis is not [], 'no inputs provided.'

    single_images_2_tfrecord(args.filename, niftis)


def generate_parser():
    parser = argparse.ArgumentParser(
        description='command line interface packs mri images into tfrecord. '
                    'The module also contains input functions for iterating '
                    'through datasets generated with this cli.'
    )
    parser.add_argument('niftis', nargs='*',
                        help='space separated list of nifti filenames.')
    parser.add_argument('--list',
                        help='text file with new line delimited files.')
    parser.add_argument('--filename', default='niftis.tfrecord',
                        help='path of tfrecord file to save out.')
    return parser


def single_images_2_tfrecord(record_name, simple_list):
    """
    writes a set of nifti file names into a tfrecord
    :param record_name: path to tfrecord
    :param simple_list: list of nifti paths
    :return:
    """
    gen = single_image_data_generator('X', simple_list)
    write_tfrecord(record_name, gen)


def write_tfrecord(filename, feature_sets, clobber=True):
    """

    :param filename: name of tf records file
    :param feed_dict: dictionary of parameter names and datasets
    :param clobber: overwrite existing tfrecord.
    :return:
    """

    label_dict = {'GE': 0, 'PHILIPS': 1, 'SIEMENS': 2}

    if clobber and os.path.exists(filename):
        os.remove(filename)

    with tf.io.TFRecordWriter(filename) as writer:
        for data_dict in feature_sets:
            # @ TODO allow for writing labels
            feature = {}
            for name, data in data_dict.items():
                if isinstance(data, np.ndarray):
                    data = data[:, :, 85]  # pick a z-slice
                    # write numpy array into tf bytes feature
                    bytess = data.tostring()
                    feature[f'{name}'] = _bytes_feature(tf.compat.as_bytes(bytess))
                elif isinstance(data, str):
                    code = label_dict[data]
                    one_hot = [0, 0, 0]
                    one_hot[code] = 1
                    feature[f'{name}'] = _int64_feature(one_hot)

            # create protocol buffer from bytes feature.
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            serialized = example.SerializeToString()
            writer.write(serialized)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def single_image_data_generator(name, files):
    for f in files:
        try:
            dat = {name: load_image(f), 'y': scanner_name.match(f).group(1)}
        except EOFError:
            continue
        yield dat


def load_image(addr):
    img = nb.load(addr)
    img_data = img.get_fdata().astype(np.float32)

    return img_data


def single_image_parser(serialized):

    features = {'X': tf.FixedLenFeature([], tf.string)}
                #'y': tf.FixedLenFeature([], tf.int64)}
    example = tf.parse_single_example(serialized=serialized,
                                      features=features)
    image_raw = example['X']
    image = tf.decode_raw(image_raw, tf.float32)
    image = tf.reshape(image, (182, 218, 1))
    m = tf.math.reduce_min(image)
    M = tf.contrib.distributions.percentile(image, q=99)
    image = (image - m) / (M - m)
    # image = tf.image.random_contrast(image, lower=0.3, upper=2.0)
    # image = tf.image.random_brightness(image, max_delta=2.0)
    #  padding = tf.constant([[37, 37], [19, 19], [0, 0]])  # pad to power of 2 (probably way inefficient)
    #  image = tf.pad(image, padding)

    return image, image


def preproc(image, *args):
    mask = tf.where(image > 0.0, tf.ones_like(image), tf.zeros_like(image))
    image = tf.image.random_contrast(image, lower=0.66, upper=1.5)
    image = image + tf.random_normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
    image = tf.image.random_flip_up_down(image)  # images are stored x = A->P
    image = image * mask  # image is skull stripped, brain is the only nonzero area
    return image, image


def image_input_fn(filenames, train, batch_size=4, buffer_size=512,
                   shuffle=True, labels=False):

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(single_image_parser)
    if train:
    #    dataset = dataset.map(preproc)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = None
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)

    # autoencoder has same input and validation
    iterator = dataset.make_one_shot_iterator()
    # iterator2 = dataset.make_one_shot_iterator()
    image_batch, image_batch2 = iterator.get_next()
    # image_batch2 = iterator2.get_next()

    if labels:
        x = {'X': image_batch, 'y': label_batch}
        y = {'Xout': image_batch2, 'yout': label_batch2}
    else:
        x = {'X': image_batch}
        y = {'Xout': image_batch2}

    return x, y


if __name__ == '__main__':
    _cli()
