#!/usr/bin/env python
import numpy as np
import os

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
def getAnnotation(filePath):
    file = open(filePath, 'r')
    data = []
    for line in file:
        data.append(line.split())
    array = np.array(data)
    array = array.astype(np.float)# since the data are frame number which is int and valence arousal are range from -1000 to 1000
    array = array.astype(np.int)
    return array
    
videoPath = '/vol/bitbucket/ml9915/train/trainVideos/'
annotationPath = '/vol/bitbucket/ml9915/train/trainLabel/'

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
        # specify the structure of your data before you write it to the file.
        feature = {
                        'frameNo': _bytes_feature(tf.compat.as_bytes(videoName+'/'+name)),
                       'frame': _bytes_feature(tf.compat.as_bytes(img.tostring())),
                      'valence': _int64_feature(record[1]),
                      'arousal': _int64_feature(record[2])
                  }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to binary string and write on the file
        # TFRecord file stores your data as a sequence of binary strings.
        writer.write(example.SerializeToString())
    writer.close()
    print 'finish ' + videoName
