#!/usr/bin/env python3
import os
import glob
import shutil
import sys

import numpy as np

import inputs
import optimization

PROJECT_ROOT = os.path.expanduser('~/cvae')
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
    
    train_record = os.path.join(TMPDIR, 'train.tfrecord')
    test_record = os.path.join(TMPDIR, 'test.tfrecord')
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'train.tfrecord')):
        # train/test split
        os.makedirs(TMPDIR, exist_ok=True)
        inputs.single_images_2_tfrecord(train_record, train_filenames)
        inputs.single_images_2_tfrecord(test_record, test_filenames)
        shutil.move(TMPDIR, PROJECT_ROOT)
    train_record = os.path.join(PROJECT_ROOT, 'train.tfrecord')
    test_record = os.path.join(PROJECT_ROOT, 'test.tfrecord')

    return train_record, test_record


def maybe_create_2D_record():
    t1w_pattern = '/home/groups/brainmri/abcd/data/HCP/processed/' \
                  'ABCD_*_SEFMNoT2/sub-*/ses-baselineYear1Arm1/' \
                  'HCP_release_20170910_v1.1/sub-*/T1w/' \
                  'T1w_acpc_dc_restore_brain.nii.gz'
    filenames = glob.glob(t1w_pattern)
    assert len(filenames)
    size = len(filenames)
    print('size of dataset = %s' % size)
    # randomize
    np.random.shuffle(filenames)
    train_filenames = filenames[:int(size * .9)]
    test_filenames = filenames[int(size * .9):]
    train_record = os.path.join(TMPDIR, 'train.tfrecord')
    test_record = os.path.join(TMPDIR, 'test.tfrecord')
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'train.tfrecord')):
        # train/test split
        os.makedirs(TMPDIR, exist_ok=True)
        inputs.single_images_2_tfrecord(train_record, train_filenames)
        inputs.single_images_2_tfrecord(test_record, test_filenames)
        shutil.move(TMPDIR, PROJECT_ROOT)
    train_record = os.path.join(PROJECT_ROOT, 'train.tfrecord')
    test_record = os.path.join(PROJECT_ROOT, 'test.tfrecord')
    return train_record, test_record


def run_hyperparameter_optimization(train_record, test_record, working_dir,
                                    n_jobs=1):
    # this may require no real prep
    optimization.hyperparameter_optimization(train_record, test_record,
                                             working_dir=working_dir, n_jobs=n_jobs)


if __name__ == '__main__':
    # train_record, test_record = maybe_create_2D_record()
    wd = os.path.join(PROJECT_ROOT, 'model') # don't do any /mnt/scratch
    train_record = os.path.join(PROJECT_ROOT, 'train.tfrecord')
    test_record = os.path.join(PROJECT_ROOT, 'test.tfrecord')
    try:
        os.makedirs(wd, exist_ok=True)
    except PermissionError:
        wd = os.path.join(PROJECT_ROOT, sys.argv[1])
        os.makedirs(wd, exist_ok=True)
    run_hyperparameter_optimization([train_record], [test_record],
                                    working_dir=wd)
