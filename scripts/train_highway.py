#!/usr/bin/env python

import tensorflow as tf
import data_utils_mean
import vgg_face
import models
import losses
from tensorflow.python.platform import tf_logging as loggingTF

import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('/vol/gpudata/ml9915/summary/face-gru-highway/output_train.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

slim = tf.contrib.slim

'''the directory to save the model checkpoints, weights and event files  '''
TRAIN_DIR = '/vol/gpudata/ml9915/check_point/face-gru-highway/'
TRAIN_TFR_PATH = '/vol/gpudata/ml9915/TFRecords/train/'
VGG_RESTORE_PATH = '/vol/atlas/homes/dk15/pami/vgg_face_restore/model.ckpt-0'

# HYPER-PARAMETERS
BATCH_SIZE = 2
SEQUENCE_LENGTH = 80
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
USE_CCC = True
WIDTH = 96
HEIGHT = 96
CHANNELS = 3

def reshape_to_cnn(images):
    image_batch = tf.reshape(images, (BATCH_SIZE * SEQUENCE_LENGTH, HEIGHT, WIDTH, CHANNELS))
    return image_batch

def reshape_to_rnn(tensor):
    tensor = tf.reshape(tensor, [BATCH_SIZE, SEQUENCE_LENGTH, -1])
    return tensor

def compute_loss(prediction, labels_batch):
    mse_total = []
    ccc_total = []
    predictions = tf.reshape(prediction, [BATCH_SIZE*SEQUENCE_LENGTH, 2])
    labels = tf.reshape(labels_batch, [BATCH_SIZE*SEQUENCE_LENGTH, 2])
    for i, name in enumerate(['valence', 'arousal']):
        pred_single = tf.reshape(predictions[:, i], (-1,))
        gt_single = tf.reshape(labels[:, i], (-1,))
        # compute ccc
        loss_ccc = losses.concordance_cc2(pred_single, gt_single)
        tf.summary.scalar('losses/CCC_loss_' + name, loss_ccc)
        ccc_total.append(loss_ccc)
        # compute mse
        loss_mse = tf.reduce_mean(tf.square(pred_single - gt_single))
        tf.summary.scalar('losses/MSE_loss_' + name, loss_mse)
        mse_total.append(loss_mse)
        if USE_CCC:
            loss = loss_ccc
        else:
            loss = loss_mse
        slim.losses.add_loss(loss / 2.)
    tf.summary.scalar('losses/CCC_loss', (ccc_total[0] + ccc_total[1])/2.0)
    tf.summary.scalar('losses/MSE_loss', (mse_total[0] + mse_total[1])/2.0)


def train():
    g = tf.Graph()
    with g.as_default():
        # load data get iterator
        data_loader = data_utils_mean.DataLoader(SEQUENCE_LENGTH, BATCH_SIZE, NUM_EPOCHS)
        iterator = data_loader.load_data(TRAIN_TFR_PATH, True)
        with tf.Session(graph=g) as sess:
            frameNo, image, label = iterator.get_next()

            # VGG FACE network
            VGGFace_network = vgg_face.VGGFace(SEQUENCE_LENGTH * BATCH_SIZE)
            image_batch = reshape_to_cnn(image)
            VGGFace_network.setup(image_batch, trainable=False)
            face_output = VGGFace_network.get_face_fc0()

            # RNN part
            rnn_in = reshape_to_rnn(face_output)
            prediction = models.get_prediction_highway(rnn_in)
            prediction = tf.reshape(prediction, [BATCH_SIZE, SEQUENCE_LENGTH, 2])

            label_batch = tf.reshape(label, [BATCH_SIZE, SEQUENCE_LENGTH, 2])

            # compute losses using slim
            compute_loss(prediction, label_batch)

            total_loss = slim.losses.get_total_loss()
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

            # restore VGG-FACE model at the beginning
            restore_names = VGGFace_network.get_restore_vars()
            variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=restore_names)
            init_fn = slim.assign_from_checkpoint_fn(VGG_RESTORE_PATH, variables_to_restore, ignore_missing_vars=False)


            # summarize_gradients : Whether or not add summaries for each gradient.
            # variables_to_train: an optional list of variables to train. If None, it will default to all tf.trainable_variables().
            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True #  Whether or not add summaries for each gradient.
                                                     )
            loggingTF.set_verbosity(1)
            # keep 10000 ckpts
            saver = tf.train.Saver(max_to_keep=10000)
            # including initialize local and global variables

            slim.learning.train(train_op,
                            TRAIN_DIR,
                            init_fn=init_fn,
                            save_summaries_secs=60 * 15, # How often, in seconds, to save summaries.
                            log_every_n_steps=500, # The frequency, in terms of global steps, that the loss and global step are logged.
                            save_interval_secs=60 * 15, # How often, in seconds, to save the model to `logdir`.
                            saver=saver
                            )



if __name__ == '__main__':
    train()
