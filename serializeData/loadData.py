import cv2
import tensorflow as tf
import numpy as np
import os

def getAnnotation(filePath):
    file = open(filePath, 'r')
    data = []
    for line in file:
        data.append(line.split())
    array = np.array(data)
    array = array.astype(np.float)# since the data are frame number which is int and valence arousal are range from -1000 to 1000
    array = array.astype(np.int)
    return array

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def getVideoNames(annotationPath):
    files = os.listdir(annotationPath)
    drop = filter(lambda x: x.startswith('.'), files)
    for item in drop:
        files.remove(item)
    videoNames = []
    for item in files:
        videoNames.append(item.split('.')[0])
    videoNames = sorted(videoNames)
    return videoNames
# TF Records features
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
videoPath = '/vol/bitbucket/ml9915/validation/validationVideos/'
annotationPath = '/vol/bitbucket/ml9915/validation/validationLabel/'
tfRecordsPath = '/vol/bitbucket/ml9915/TFRecords/validation/'

videoNames = getVideoNames(annotationPath)
for videoName in videoNames:
    # get labels
    labelFile = annotationPath+videoName+'.txt'
    # frameNo, Valence, Arousal in one video
    frameVA = getAnnotation(labelFile)
    tfRecordFilePath = tfRecordsPath+videoName+'.tfrecords'
    framesPath = videoPath+videoName+'/'
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(tfRecordFilePath)
    for record in frameVA:
        name = '{0:05d}'.format(record[0])
        img = load_image(framesPath+name+'.jpg')
        # Create a feature
        feature = {
                        'frameNo': _bytes_feature(tf.compat.as_bytes(name)),
                       'frame': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                      'valence': _int64_feature(record[1]),
                      'arousal': _int64_feature(record[2])
                  }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    print 'finish ' + videoName
