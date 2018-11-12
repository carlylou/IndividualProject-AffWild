#!/usr/bin/env python

import tensorflow as tf
import data_utils
import vgg_face
import models
slim = tf.contrib.slim

'''the directory to save the model checkpoints, weights and event files  '''
VALID_TFR_PATH = '/vol/gpudata/ml9915/TFRecords/validation/'
CHECK_POINT_DIR = '/vol/gpudata/ml9915/check_point/gru/'
SUMMARY_PATH = '/vol/gpudata/ml9915/summary/gru/test/'
# HYPER-PARAMETERS
BATCH_SIZE = 1
SEQUENCE_LENGTH = 80
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
WIDTH = 96
HEIGHT = 96
CHANNELS = 3
FRAMES_VALID = 94223
NUM_BATCHES = FRAMES_VALID / (SEQUENCE_LENGTH * BATCH_SIZE)

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
    model_path = '/vol/gpudata/ml9915/check_point/gru/model.ckpt-242479'
    saver.restore(sess, model_path)
    step = int(model_path.split('-')[1])
    print('Variables restored from [{}]'.format(model_path))
    return step

def evaluate():
    g = tf.Graph()
    with g.as_default():
        # load data get iterator
        data_loader = data_utils.DataLoader(SEQUENCE_LENGTH, BATCH_SIZE, NUM_EPOCHS)
        iterator = data_loader.load_data(VALID_TFR_PATH, False)
        frameNo, image, label = iterator.get_next()
        # VGG FACE network
        VGGFace_network = vgg_face.VGGFace(SEQUENCE_LENGTH * BATCH_SIZE)
        image_batch = tf.reshape(image, [-1, 96, 96, 3])
        VGGFace_network.setup(image_batch)  # image_batch is a tensor of shape (batch_size*seq_length,image_dim,image_dim,3)
        face_output = VGGFace_network.get_face_fc0()
        # RNN part
        rnn_in = reshape_to_rnn(face_output)
        prediction = models.get_prediction(rnn_in)
        prediction = tf.reshape(prediction, [-1, 2])
        label_batch = tf.reshape(label, [-1, 2])
        with tf.Session(graph=g) as sess:
            # if not os.path.exists(SUMMARY_PATH):
            #     os.makedirs(SUMMARY_PATH)
            # eval_summary_writer = tf.summary.FileWriter(SUMMARY_PATH, graph=g)
            restore_variables(sess)
            total_ccc_v = 0
            total_ccc_a = 0
            total_ccc = 0
            total_mse_v = 0
            total_mse_a = 0
            total_mse = 0
            for i in range(NUM_BATCHES):
                try :
                    pred, lab = sess.run([prediction, label_batch])
                except tf.errors.OutOfRangeError:
                    break
                print 'prediction batch : ' + str(i)
                print pred
                conc_arousal = concordance_cc2(pred[:, 1], lab[:, 1])
                conc_valence = concordance_cc2(pred[:, 0], lab[:, 0])
                mse_arousal = sum((pred[:, 1] - lab[:, 1]) ** 2) / len(lab[:, 1])
                mse_valence = sum((pred[:, 0] - lab[:, 0]) ** 2) / len(lab[:, 0])
                total_ccc_v += conc_valence
                total_ccc_a += conc_arousal
                total_ccc += ((conc_valence + conc_valence)/2.0)
                total_mse_v += (mse_valence)
                total_mse_a += (mse_arousal)
                total_mse += ((mse_arousal+mse_valence)/2.0)
            print 'Finish read data'
            # add summary
            num_batches = float(NUM_BATCHES)
            # summary = tf.Summary()
            # summary.value.add(tag='eval/conc_valence', simple_value=float(total_ccc_v/num_batches))
            # summary.value.add(tag='eval/conc_arousal', simple_value=float(total_ccc_a/num_batches))
            # summary.value.add(tag='eval/conc_total', simple_value=float(total_ccc/num_batches))
            # summary.value.add(tag='eval/mse_arousal', simple_value=float(total_mse_a/num_batches))
            # summary.value.add(tag='eval/mse_valence', simple_value=float(total_mse_v/num_batches))
            #
            # eval_summary_writer.add_summary(summary, get_global_step(model_path))

            print '#####################Summary#######################'
            # print 'Evaluate model {}'.format(model_path)
            print 'Concordance on valence : {}'.format(float(total_ccc_v/num_batches))
            print 'Concordance on arousal : {}'.format(float(total_ccc_a/num_batches))
            print 'Concordance on total : {}'.format(float(total_ccc/num_batches))
            print 'MSE Arousal : {}'.format(float(total_mse_a/num_batches))
            print 'MSE Valence : {}'.format(float(total_mse_v/num_batches))
            print 'MSE TOTAL : {}'.format(float(total_mse/num_batches))
            # eval_summary_writer.close()

if __name__ == '__main__':
    evaluate()
