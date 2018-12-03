#!/usr/bin/env python3
import argparse
import os

import nibabel as nb
import numpy as np
import tensorflow as tf


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
    if clobber and os.path.exists(filename):
        os.remove(filename)

    with tf.io.TFRecordWriter(filename) as writer:
        for data_dict in feature_sets:
            # @ TODO allow for writing labels
            feature = {}
            for name, data in data_dict.items():
                # write numpy array into tf bytes feature
                bytess = data.tobytes()
                feature[f'{name}'] = _bytes_feature(tf.compat.as_bytes(bytess))

            # create protocol buffer from bytes feature.
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            serialized = example.SerializeToString()
            writer.write(serialized)


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def single_image_data_generator(name, files):
    for f in files:
        yield {name: load_image(f)}


def load_image(addr):
    img = nb.load(addr)
    img_data = img.get_fdata().astype(np.float32)

    return img_data


def single_image_parser(serialized):

    features = {'X': tf.FixedLenFeature([], tf.string)}
    example = tf.parse_single_example(serialized=serialized,
                                      features=features)
    image_raw = example['X']
    image = tf.decode_raw(image_raw, tf.float32)
    image = tf.reshape(image, (176, 256, 256, 1))

    return image


def image_input_fn(filenames, train, batch_size=32, buffer_size=2048,
                          shuffle=True):

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(single_image_parser)
    if train:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        num_repeat = None
    else:
        num_repeat = 1
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    image_batch = iterator.get_next()

    x = {'X': image_batch}

    return x


if __name__ == '__main__':
    _cli()
