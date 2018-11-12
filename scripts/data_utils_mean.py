#!/usr/bin/env python

import os
import tensorflow as tf


class DataLoader(object):

    def __init__(self, seq_length, batch_size, epochs):
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.buffer_size = 10 * self.batch_size * self.seq_length
        self.threshold = 15 * self.seq_length

    # given tfRecordsPath return tfrecord files
    def getTFRecordsFiles(self, tfRecordsPath):
        files = os.listdir(tfRecordsPath)
        drop = filter(lambda x: x.startswith('.'), files)
        for item in drop:
            files.remove(item)
        return files

    def getTFRecordsPathList(self, tfRecordsPath):
        tfRecordsFiles = self.getTFRecordsFiles(tfRecordsPath)
        # read separate video data
        argumentsFiles = []
        for filename in tfRecordsFiles:
            argumentsFiles.append(tfRecordsPath + filename)
        return argumentsFiles

    # check a sequence consecutive
    def _check_consecutive(self, frameNo, image, label):
        firstPath = tf.string_split([frameNo[0]], '/')
        lastPath = tf.string_split([frameNo[-1]], '/')
        # if frames are not belong to same video, return false
        firstVideo = firstPath.values[0]
        lastVideo = lastPath.values[0]
        compareVideoName = tf.equal(firstVideo, lastVideo)
        # if frames have gap larger than threshold, then return False
        firstFrame = firstPath.values[1]
        lastFrame = lastPath.values[1]
        firstFrame = tf.string_to_number(firstFrame, out_type=tf.int32)
        lastFrame = tf.string_to_number(lastFrame, out_type=tf.int32)
        withinGAP = tf.less(tf.subtract(lastFrame, firstFrame), [self.threshold])
        return tf.logical_and(compareVideoName, withinGAP)[0]

    # Transforms a scalar string `example_proto` into a pair of a scalar string and
    # a scalar integer, representing an image and its label, respectively.
    def _parse_function(self, serialized_example):
        feature = {
            'frameNo': tf.FixedLenFeature([], tf.string),
            'frame': tf.FixedLenFeature([], tf.string),
            'valence': tf.FixedLenFeature([], tf.int64),
            'arousal': tf.FixedLenFeature([], tf.int64)
        }
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['frame'], tf.float32)
        # Reshape image data into the original shape
        image = tf.reshape(image, [96, 96, 3])
        # decode_raw output type do not have tf.string, only have some numerical type
        frameNo = tf.cast(features['frameNo'], tf.string)
        # Cast label data into int32
        valence = tf.cast(features['valence'], tf.float32)
        arousal = tf.cast(features['arousal'], tf.float32)
        # Preprocessing here ...
        # VA value convert to [-1, 1] tensor operation
        valence /= 1000.0
        arousal /= 1000.0

        # preprocessing for vgg face
        image = self.scale(image)
        label = tf.stack([valence, arousal], axis=0)
        return frameNo, image, label

    def scale(self, image):
        # preprocessing for vgg face
        red, green, blue = tf.split(image, num_or_size_splits=3, axis=2)
        assert red.get_shape().as_list() == [96, 96, 1]
        assert green.get_shape().as_list() == [96, 96, 1]
        assert blue.get_shape().as_list() == [96, 96, 1]
        image = tf.concat([
            blue,
            green,
            red,
        ], axis=2)
        assert image.get_shape().as_list() == [96, 96, 3]
        image -= 128.0
        image /= 128.0
        return image

    # def subtract_mean(self, image):
    #     # preprocessing for vgg face
    #     VGG_FACE_MEAN = [129.1863, 104.7624, 93.5940]  # This is B-G-R for VGG-FACE
    #     red, green, blue = tf.split(image, num_or_size_splits=3, axis=2)
    #
    #     assert red.get_shape().as_list() == [96, 96, 1]
    #     assert green.get_shape().as_list() == [96, 96, 1]
    #     assert blue.get_shape().as_list() == [96, 96, 1]
    #
    #     image = tf.concat([
    #         blue - VGG_FACE_MEAN[0],
    #         green - VGG_FACE_MEAN[1],
    #         red - VGG_FACE_MEAN[2],
    #     ], axis=2)
    #     assert image.get_shape().as_list() == [96, 96, 3]
    #     # image /= 129.1863
    #     return image

    def load_data(self, tfRecordsPath, is_training):
        # use Dataset API to read TFRecords:
        dataset = tf.data.TFRecordDataset(self.getTFRecordsPathList(tfRecordsPath))
        dataset = dataset.map(self._parse_function)  # Parse the record into tensors.
        # iterate over a dataset in multiple epochs
        if is_training:
            dataset = dataset.repeat() #indefinitely
        else:
            dataset = dataset.repeat(self.num_epochs)
        # if the batch size does not evenly divide the input dataset size, drop the final smaller element
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.seq_length))
        dataset = dataset.filter(self._check_consecutive)
        # it cost too long, comment out when test
        if is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        iterator = dataset.make_one_shot_iterator()
        return iterator