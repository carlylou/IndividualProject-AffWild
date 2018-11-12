#!/usr/bin/env python

import tensorflow as tf
import data_utils
import vgg_face
import models
import losses
import time

from tensorflow.python.platform import tf_logging as loggingTF

import logging

# get TF logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create file handler which logs even debug messages
fh = logging.FileHandler('/vol/gpudata/ml9915/summary/gru-debug/output_train.txt')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

slim = tf.contrib.slim

'''the directory to save the model checkpoints, weights and event files  '''
TRAIN_DIR = '/vol/gpudata/ml9915/check_point/gru-debug/'
TRAIN_TFR_PATH = '/vol/gpudata/ml9915/TFRecords/train/'
VGG_RESTORE_PATH = '/vol/atlas/homes/dk15/pami/vgg_face_restore/model.ckpt-0'
OUTPUT_FILE = '/vol/gpudata/ml9915/summary/gru-debug/output_train.txt'
# "/vol/gpudata/ml9915/check_point/gru-debug/model.ckpt-8511"
MODEL_PATH = TRAIN_DIR + "model.ckpt"
# HYPER-PARAMETERS
BATCH_SIZE = 5
SEQUENCE_LENGTH = 80
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
USE_CCC = True
WIDTH = 96
HEIGHT = 96
CHANNELS = 3
FRAMES_VALID = 527056

MAX_STEP = NUM_EPOCHS * (FRAMES_VALID / (BATCH_SIZE * SEQUENCE_LENGTH))

def write(content):
    file = open(OUTPUT_FILE, "a")
    file.write(content)
    file.close()

def reshape_to_cnn(images):
    image_batch = tf.reshape(images, (BATCH_SIZE * SEQUENCE_LENGTH, HEIGHT, WIDTH, CHANNELS))
    return image_batch

def reshape_to_rnn(tensor):
    tensor = tf.reshape(tensor, [BATCH_SIZE, SEQUENCE_LENGTH, -1])
    return tensor

def compute_loss(prediction, labels_batch):
    with tf.name_scope('compute_loss'):
        mse_total = []
        ccc_total = []
        predictions = tf.reshape(prediction, [BATCH_SIZE*SEQUENCE_LENGTH, 2])
        labels = tf.reshape(labels_batch, [BATCH_SIZE*SEQUENCE_LENGTH, 2])
        for i, name in enumerate(['valence', 'arousal']):
            pred_single = predictions[:, i]
            gt_single = labels[:, i]
            # compute ccc
            loss_ccc = losses.concordance_cc2(pred_single, gt_single)
            ccc_total.append(loss_ccc)
            # compute mse
            loss_mse = tf.reduce_mean(tf.square(pred_single - gt_single))
            mse_total.append(loss_mse)
            if USE_CCC:
                loss = loss_ccc
            else:
                loss = loss_mse
            slim.losses.add_loss(loss / 2.)
        tf.summary.scalar('losses/CCC_loss', (ccc_total[0] + ccc_total[1])/2.0)
        tf.summary.scalar('losses/MSE_loss', (mse_total[0] + mse_total[1])/2.0)

# return global step
def restore_all_variables(sess, VGGFace_network):
    model_path = tf.train.latest_checkpoint(TRAIN_DIR)
    if model_path != None:
        # Add ops to save and restore only `v2` using the name "v2"
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        step = int(model_path.split('-')[1])
        write('Variables restored from [{}]'.format(model_path))
        return step
    else:
        # init global and local variables and load weights from pretrained-model
        # restore VGG-FACE model at the beginning
        restore_names = VGGFace_network.get_restore_vars()
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=restore_names)
        # Initialize all global and local variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Add ops to save and restore only `v2` using the name "v2"
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(VGG_RESTORE_PATH, variables_to_restore)
        init_fn(sess) # load the pretrained weights
        return 0


def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op

def get_prediction(x):
    # VGG FACE network
    VGGFace_network = vgg_face.VGGFace(SEQUENCE_LENGTH * BATCH_SIZE)
    image_batch = reshape_to_cnn(x)
    VGGFace_network.setup(image_batch) # image_batch is a tensor of shape (batch_size*seq_length,image_dim,image_dim,3)
    face_output = VGGFace_network.get_face_fc0()

    # RNN part
    rnn_in = reshape_to_rnn(face_output)
    prediction = models.get_prediction(rnn_in)
    prediction = tf.reshape(prediction, [BATCH_SIZE, SEQUENCE_LENGTH, 2])
    return prediction

def get_loss(prediction, y_):
    label_batch = tf.reshape(y_, [BATCH_SIZE, SEQUENCE_LENGTH, 2])
    # compute losses using slim
    compute_loss(prediction, label_batch)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('losses/total_loss', total_loss)
    return total_loss



def train():
    g = tf.Graph()
    with g.as_default():
        with tf.Session(graph=g) as sess:
            # build the graph
            x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH, HEIGHT, WIDTH, CHANNELS])
            y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, SEQUENCE_LENGTH, 2])
            prediction = get_prediction(x)

            total_loss = get_loss(prediction, y_)

            my_global_step = tf.get_variable("global_step", initializer=tf.constant(0), trainable=False)
            # restore or init variables
            step = restore_all_variables(sess)
            # ???????????????????????????
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss, global_step=step)
            # prepare summary & saver
            saver_model = tf.train.Saver(max_to_keep=10000)
            summary_writer = tf.summary.FileWriter(TRAIN_DIR, graph=sess.graph)
            merged_summaries = tf.summary.merge_all()
            # get data iterator
            data_loader = data_utils.DataLoader(SEQUENCE_LENGTH, BATCH_SIZE, NUM_EPOCHS)
            iterator = data_loader.load_data(TRAIN_TFR_PATH, True)
            frameNo, image, label = iterator.get_next()
            for i in range(MAX_STEP):
                try:
                    start_time = time.time()
                    frameNos, images, labels = sess.run([frameNo, image, label])
                    _, step_loss, step_summary = sess.run([optimizer], feed_dict={x:images, y_:labels})
                    time_step = time.time() - start_time
                    if step % 500 == 0 or (step + 1) == MAX_STEP:
                        step_loss, step_summary = sess.run([total_loss, merged_summaries], feed_dict={x:images, y_:labels})
                        summary_writer.add_summary(step_summary, global_step=step)
                        write("Global_Step {}: loss = {:.4f} ({:.2f} sec/step)".format(step, step_loss, time_step))

                    if step % 1500 == 0 or (step + 1) == MAX_STEP:
                        # save model to ckpts
                        saver_model.save(sess, MODEL_PATH, global_step=step)
                    step += 1
                except tf.errors.OutOfRangeError:
                    break
            summary_writer.close()
        sess.close()




if __name__ == '__main__':
    train()
