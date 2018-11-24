#!/usr/bin/env python3
import os

import inputs

PROJECT_ROOT = '/home/exacloud/lustre1/fnl_lab/projects/darrick_cnn'
NIFTI_SRC_ROOT = ''
ESTIMATOR_FOLDER = os.path.join(PROJECT_ROOT, 'vae_model')


def maybe_create_record():
    train_record = os.path.join(PROJECT_ROOT, 'train.tfrecord')
    test_record = os.path.join(PROJECT_ROOT, 'test.tfrecord')
    if not os.path.exists(train_record):
        # get all niftis...
        # train/test split
        inputs.single_images_2_tfrecord(train_record)
        inputs.single_images_2_tfrecord(test_record)

    return train_record, test_record


def run_model(train_record):
    pass


if __name__ == '__main__':
    maybe_create_record()

