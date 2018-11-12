#!/usr/bin/env python

import cv2 as cv
import cv2
import numpy as np
import scipy
from scipy.misc import imread
import random
import os
import matplotlib.pyplot as plt
import subprocess

def getHist(imagePath):
    # extract a 3D RGB color histogram from the image,
    # using 8 bins per channel, normalize, and update
    # the index
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compareHist(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_HELLINGER)

# Feature extractor
def extract_features(image_path, vector_size=32):
    image = imread(image_path, mode="RGB")
    try:
        # SIFT_create()
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        if dsc is None:
            return None
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except:
        print 'Error: '+image_path
        return None
    return dsc

def cos_cdist(vector1, vector2):
    # getting cosine distance between search image and images database
    vector2 = vector2.reshape(1, -1)
    vector1 = vector1.reshape(1, -1)
    return scipy.spatial.distance.cdist(vector1, vector2, 'cosine').reshape(-1)

def startWithNumber(number, frameList):
    name = '{0:05d}'.format(number)
    result = filter(lambda x: x.startswith(name), frameList)
    return result

pathOut = ('/vol/bitbucket/ml9915/filterSimilarity/')
pathIn = ('/vol/bitbucket/ml9915/ffcropped1/')
files = ['6-30-1920x1080']

for filename in files:
    print filename
    framePath = pathIn+filename+'/'
    frames = os.listdir(framePath)
    frames = sorted(frames)
    # get min and max frame number
    start = frames[0].split('.')[0].split('-')[0]
    end = frames[-1].split('.')[0].split('-')[0]

    outputFramePath1 = pathOut+filename+'-p1/'
    if not os.path.exists(outputFramePath1):
        os.makedirs(outputFramePath1)
    outputFramePath2 = pathOut+filename+'-p2/'
    if not os.path.exists(outputFramePath2):
        os.makedirs(outputFramePath2)
    # iterate all frame number and select the only one detected face for that frame
    res = startWithNumber(int(start), frames)
    res = sorted(res)
    # get target feature / histogram
    targetFeature1 = extract_features(framePath+res[0])
    targetHist1 = getHist(framePath+res[0])
    # targetFeature2 = extract_features(framePath+res[1])
    # targetHist2 = getHist(framePath+res[1])

    name = res[0].split('.')[0].split('-')[0]

    subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath1+name+'.jpg', shell=True)
    subprocess.call('cp ' + framePath+res[1] + ' '+outputFramePath2+name+'.jpg', shell=True)
    for i in range(int(start)+1, int(end)+1):
        res = startWithNumber(i, frames)
        res = sorted(res)
        if len(res) == 0:
            continue
        elif len(res) == 1:
            tempFeature = extract_features(framePath+res[0])
            # tempHist = getHist(framePath+res[0])
            if tempFeature is not None:
                d1 = cos_cdist(targetFeature1, tempFeature)
                # d2 = cos_cdist(targetFeature2, tempFeature)
            else:
                d1 = 0
                # d2 = 0
            name = res[0].split('.')[0].split('-')[0]
            if (d1 < 0.5):
                if tempFeature is not None:
                    targetFeature1 = tempFeature
                # targetHist1 = tempHist
                subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath1+name+'.jpg', shell=True)
            else :
                # if tempFeature is not None:
                #     targetFeature2 = tempFeature
                # targetHist2 = tempHist
                subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath2+name+'.jpg', shell=True)
        else:
            # len(res) >1
            disDict1 = {}
            # disDict2 = {}
            for imgName in res:
                tempFeature = extract_features(framePath + imgName)
                if tempFeature is None:
                    continue
                tempD1 = cos_cdist(targetFeature1, tempFeature)
                # tempD2 = cos_cdist(targetFeature2, tempFeature)
                disDict1[str(tempD1)] = imgName
                # disDict2[str(tempD2)] = imgName
            keys1 = sorted(disDict1.keys())
            # keys2 = sorted(disDict2.keys())
            if len(keys1) < 1:
                continue
            target1 = disDict1[keys1[0]]
            # target2 = disDict2[keys2[0]]
            # targetHist1 = getHist(framePath + target)
            targetFeature1 = extract_features(framePath + target1)
            # targetFeature2 = extract_features(framePath + target2)
            name = target1.split('.')[0].split('-')[0]
            subprocess.call('cp ' + framePath+target1 + ' '+outputFramePath1+name+'.jpg', shell=True)
            if len(keys1) < 2:
                continue
            subprocess.call('cp ' + framePath+disDict1[keys1[1]] + ' '+outputFramePath2+name+'.jpg', shell=True)
