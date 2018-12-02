#!/usr/bin/env python3
import os
import glob
import shutil

import numpy as np

import inputs

PROJECT_ROOT = '/home/exacloud/lustre1/fnl_lab/projects/darrick_cnn'
NIFTI_SRC_ROOT = '/home/exacloud/lustre1/fnl_lab'
ESTIMATOR_FOLDER = os.path.join(PROJECT_ROOT, 'vae_model')
TMPDIR = '/mnt/scratch/darrick_cnn'


def maybe_create_record():
    t1w_pattern = '/home/exacloud/lustre1/fnl_lab/data/HCP/sorted/ABCD' \
                  '/sub-*/ses-*/anat/*T1w.nii.gz'
    filenames = glob.glob(t1w_pattern)
    size = len(filenames)
    print('size of dataset = %s' % size)

    # randomize
    np.random.shuffle(filenames)

    train_filenames = filenames[:int(size * .9)]
    test_filenames = filenames[int(size * .9):]

    os.makedirs(TMPDIR, exist_ok=True)
    train_record = os.path.join(TMPDIR, 'train.tfrecord')
    test_record = os.path.join(TMPDIR, 'test.tfrecord')
    if not os.path.exists(train_record):
        # get all niftis...
        # train/test split
        inputs.single_images_2_tfrecord(train_record, train_filenames)
        inputs.single_images_2_tfrecord(test_record, test_filenames)

    #shutil.move(TMPDIR, PROJECT_ROOT)

    return train_record, test_record


def run_model():
    pass


def run_hyperparameter_optimization():
    pass


if __name__ == '__main__':
    maybe_create_record()
