import argparse
import os

import tensorflow as tf
import nibabel as nb
import numpy as np


def _cli():
    pass


def single_images_2_tfrecord(record_name, simple_list):
    """
    writes a set of nifti file names into a tfrecord
    :param record_name: path to tfrecord
    :param simple_list: list of nifti paths
    :return:
    """
    gen = single_image_data_generator(simple_list)
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

    writer = tf.io.TFRecordWriter(filename)

    def write_to_record(data_dict):
        """
        writes numpy data arrays into tfrecord features.
        :param data_dict: dict of numpy arrays and names
        :return:
        """
        # @ TODO allow for writing labels
        feature = {}
        for name, data in data_dict:
            # write numpy array into tf bytes feature
            feature[f'{name}'] = _bytes_feature(tf.compat.as_bytes(data.tostring()))

        # create protocol buffer from bytes feature.
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    for dct in feature_sets:
        write_to_record(dct)

    writer.close()


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def single_image_data_generator(name, files):
    for f in files:
        yield {name: load_image(f)}


def load_image(addr):
    img = nb.load(addr)
    img_data = img.get_fdata().astype(np.float32)

    return img_data


if __name__ == '__main__':
    _cli()
