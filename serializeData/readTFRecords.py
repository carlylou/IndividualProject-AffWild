#!/usr/bin/env python

import cv2
import tensorflow as tf
import numpy as np
import os

# HYPER-PARAMETERS
NUM_EPOCHS = 1
BATCH_SIZE = 100


# given tfRecordsPath return tfrecord files
def getTFRecordsFiles(tfRecordsPath):
    files = os.listdir(tfRecordsPath)
    drop = filter(lambda x: x.startswith('.'), files)
    for item in drop:
        files.remove(item)
    return files

# given list of tfrecords files, return ??????
def read_from_tfrecord(tfRecordsFile):
    # Created a feature
    feature = {
                    'frameNo': tf.FixedLenFeature([], tf.string),
                   'frame': tf.FixedLenFeature([], tf.string),
                  'valence': tf.FixedLenFeature([], tf.int64),
                  'arousal': tf.FixedLenFeature([], tf.int64)
              }
    # Create a queue (filename_queue) to hold filenames:
    # string_input_producer argument : capacity: An integer. Sets the queue capacity. shuffle's default value is True
    # shuffling the filenames within an epoch if shuffle=True
    filename_queue = tf.train.string_input_producer([tfRecordsFile], name = 'queue', num_epochs=NUM_EPOCHS, shuffle=True)
    reader = tf.TFRecordReader()
    # the reader returns the next record using: reader.read(filename_queue)
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)
    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['frame'], tf.float32)
    # Reshape image data into the original shape
    image = tf.reshape(image, [96, 96, 3])
    frameNo = features['frameNo']
    # Cast label data into int32
    valence = tf.cast(features['valence'], tf.int32)
    arousal = tf.cast(features['arousal'], tf.int32)
    # Any preprocessing here ...

    # VA value convert to [-1, 1]
    rangeV = tf.fill(tf.shape(valence), 1000.0)
    valence = tf.divide(valence, rangeV)
    rangeA = tf.fill(tf.shape(arousal), 1000.0)
    arousal = tf.divide(arousal, rangeA)
    # valence = valence/1000.0
    # arousal = arousal/1000.0
    # check frame number? But how?

    # subtract mean value to images:? VGG FACE?
    image = tf.cast(image, tf.uint8)
    # Creates batches by randomly shuffling tensors
    # images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    frameNo, image, valence, arousal = tf.train.batch([frameNo, image, valence, arousal], batch_size=BATCH_SIZE, num_threads=1)
    return frameNo, image, valence, arousal


# get list of filenames
def input_pipeline():
    tfRecordsPath = '/vol/bitbucket/ml9915/TFRecords/validation/'
    tfRecordsFiles = getTFRecordsFiles(tfRecordsPath)
    # read separate video data
    for filename in tfRecordsFiles:
        frameNo, image, valence, arousal = read_from_tfrecord(tfRecordsPath+filename)



def main():
    sess = tf.Session()
    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # have nor been created yet
    train_op =
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            # Run training steps or whatever
            # train_op should use data from input_pipeline()
            sess.run(train_op)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
