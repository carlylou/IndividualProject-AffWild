#!/usr/bin/env python

import tensorflow as tf
import data_utils_mean
import vgg_face
import models
import numpy as np
slim = tf.contrib.slim

'''the directory to save the model checkpoints, weights and event files  '''
TEST_TFR_PATH = '/vol/gpudata/ml9915/TFRecords/test/'
CHECK_POINT_DIR = '/vol/gpudata/ml9915/check_point/face_atten_case3_70/'
MODEL_PATH = CHECK_POINT_DIR + 'model.ckpt-9165'
SUMMARY_PATH = '/vol/gpudata/ml9915/summary/face_atten_case3_70/test/'
OUTPUT_FILE = '/vol/gpudata/ml9915/summary/face_atten_case3_70/test/output_all_test.txt'
# HYPER-PARAMETERS
SEQUENCE_LENGTH = 70
BATCH_SIZE = 1

NUM_EPOCHS = 1

WIDTH = 96
HEIGHT = 96
CHANNELS = 3

def reshape_to_rnn(tensor):
    tensor = tf.reshape(tensor, [BATCH_SIZE, SEQUENCE_LENGTH, -1])
    return tensor

def get_global_step(model_path):
    return int(model_path.split('-')[1])

def concordance_cc2(r1, r2):
    mean_cent_prod = ((r1 - r1.mean()) * (r2 - r2.mean())).mean()
    return (2 * mean_cent_prod) / (r1.var() + r2.var() + (r1.mean() - r2.mean()) ** 2)

def restore_variables(sess):
    saver = tf.train.Saver()
    model_path = MODEL_PATH
    saver.restore(sess, model_path)
    step = int(model_path.split('-')[-1])
    print 'Variables restored from [{}]'.format(model_path)
    return step

def write(content):
    file = open(OUTPUT_FILE, "a")
    file.write(content)
    file.close()

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
        VGGFace_network.setup(image_batch)  # image_batch is a tensor of shape (batch_size*seq_length,image_dim,image_dim,3)
        face_output = VGGFace_network.get_face_fc0()
        # RNN part
        rnn_in = reshape_to_rnn(face_output)
        prediction = models.get_prediction_atten(rnn_in, attn_length=30)

        prediction = tf.reshape(prediction, [-1, 2])
        label_batch = tf.reshape(label, [-1, 2])
        with tf.Session(graph=g) as sess:
            restore_variables(sess)
            evaluated_predictions = []
            evaluated_labels = []
            while True:
                try:
                    pred, lab = sess.run([prediction, label_batch])
                    evaluated_predictions.append(pred)
                    evaluated_labels.append(lab)
                except tf.errors.OutOfRangeError:
                    break
            print 'Finish read data'
            predictions = np.reshape(evaluated_predictions, (-1, 2))
            labels = np.reshape(evaluated_labels, (-1, 2))
            for i, name in enumerate(['valence', 'arousal']):
                result = np.stack((predictions[:, i], labels[:, i]), axis=-1)
                np.savetxt(SUMMARY_PATH+name+'_test.txt', result, fmt="%.3f")
            conc_arousal = concordance_cc2(predictions[:, 1], labels[:, 1])
            conc_valence = concordance_cc2(predictions[:, 0], labels[:, 0])
            mse_arousal = sum((predictions[:, 1] - labels[:, 1]) ** 2) / len(labels[:, 1])
            mse_valence = sum((predictions[:, 0] - labels[:, 0]) ** 2) / len(labels[:, 0])
            print '#####################Summary#######################'
            print 'Concordance on valence : {}'.format(conc_valence)
            print 'Concordance on arousal : {}'.format(conc_arousal)
            print 'Concordance on total : {}'.format((conc_arousal + conc_valence) / 2)
            print 'MSE Arousal : {}'.format(mse_arousal)
            print 'MSE Valence : {}'.format(mse_valence)
            # eval_summary_writer.close()

if __name__ == '__main__':
    evaluate()
