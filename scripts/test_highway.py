#!/usr/bin/env python

import tensorflow as tf
import data_utils_mean
import vgg_face
import models
import metrics
import os
from tensorflow.python.platform import tf_logging as loggingTF
import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('/vol/gpudata/ml9915/summary/face-gru-residual/test/output_test.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

slim = tf.contrib.slim

'''the directory to save the model checkpoints, weights and event files  '''
TEST_TFR_PATH = '/vol/gpudata/ml9915/TFRecords/test/'
CHECK_POINT_DIR = '/vol/gpudata/ml9915/check_point/face-gru-residual/'
MODEL_PATH = CHECK_POINT_DIR + 'model.ckpt-55972'
SUMMARY_PATH = '/vol/gpudata/ml9915/summary/face-gru-residual/test/'
# HYPER-PARAMETERS
FRAMES_VALID = 135145
SEQUENCE_LENGTH = 80
BATCH_SIZE = 1

NUM_BATCHES = FRAMES_VALID / (SEQUENCE_LENGTH * BATCH_SIZE)

NUM_EPOCHS = 1
EVAL_INTERVAL_SECS = 300


LEARNING_RATE = 0.001
WIDTH = 96
HEIGHT = 96
CHANNELS = 3

def reshape_to_rnn(tensor):
    tensor = tf.reshape(tensor, [BATCH_SIZE, SEQUENCE_LENGTH, -1])
    return tensor

def get_global_step(model_path):
    return int(model_path.split('-')[1])



def evaluate():
    g = tf.Graph()
    with g.as_default():
        # load data get iterator
        data_loader = data_utils_mean.DataLoader(SEQUENCE_LENGTH, BATCH_SIZE, NUM_EPOCHS)
        iterator = data_loader.load_data(TEST_TFR_PATH, False)
        frameNo, image, label = iterator.get_next()
        # define model graph
        # VGG FACE network
        VGGFace_network = vgg_face.VGGFace(SEQUENCE_LENGTH * BATCH_SIZE)
        image_batch = tf.reshape(image, [-1, 96, 96, 3])
        VGGFace_network.setup(image_batch, trainable=False)  # image_batch is a tensor of shape (batch_size*seq_length,image_dim,image_dim,3)
        face_output = VGGFace_network.get_face_fc0()
        # RNN part
        rnn_in = reshape_to_rnn(face_output)
        prediction = models.get_prediction_highway(rnn_in, useHighway=False)
        prediction = tf.reshape(prediction, [BATCH_SIZE, SEQUENCE_LENGTH, 2])
        label_batch = tf.reshape(label, [BATCH_SIZE, SEQUENCE_LENGTH, 2])

        # Computing MSE and Concordance values, and adding them to summary
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'eval/mse_valence': slim.metrics.streaming_mean_squared_error(prediction[:, :, 0], label_batch[:, :, 0]),
            'eval/mse_arousal': slim.metrics.streaming_mean_squared_error(prediction[:, :, 1], label_batch[:, :, 1]),
        })

        summary_ops = []
        conc_total = 0
        mse_total = 0
        for i, name in enumerate(['valence', 'arousal']):
            with tf.name_scope(name) as scope:
                concordance_cc2, values, updates = metrics.concordance_cc2(
                    tf.reshape(prediction[:, :, i], [-1]),
                    tf.reshape(label_batch[:, :, i], [-1]))
                for n, v in updates.items():
                    names_to_updates[n + '/' + name] = v
            op = tf.summary.scalar('eval/concordance_' + name, concordance_cc2)
            op = tf.Print(op, [concordance_cc2], 'eval/concordance_' + name)
            summary_ops.append(op)

            mse_eval = 'eval/mse_' + name
            op = tf.summary.scalar(mse_eval, names_to_values[mse_eval])
            op = tf.Print(op, [names_to_values[mse_eval]], mse_eval)
            summary_ops.append(op)

            mse_total += names_to_values[mse_eval]
            conc_total += concordance_cc2
        conc_total = conc_total / 2
        mse_total = mse_total / 2

        op = tf.summary.scalar('eval/concordance_total', conc_total)
        op = tf.Print(op, [conc_total], 'eval/concordance_total')
        summary_ops.append(op)

        op = tf.summary.scalar('eval/mse_total', mse_total)
        op = tf.Print(op, [mse_total], 'eval/mse_total')
        summary_ops.append(op)

        num_batches = int(NUM_BATCHES)
        loggingTF.set_verbosity(1)
        if not os.path.exists(SUMMARY_PATH):
            os.makedirs(SUMMARY_PATH)
        # always check latest ckpt and wait for next.
        slim.evaluation.evaluate_once('',
                                      MODEL_PATH,
                                      SUMMARY_PATH,
                                      num_evals=num_batches,
                                      eval_op=list(names_to_updates.values()),
                                      summary_op=tf.summary.merge(summary_ops),
                                      )

if __name__ == '__main__':
    evaluate()
