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
files = ['5-60-1920x1080-1', '5-60-1920x1080-2', '5-60-1920x1080-3', '5-60-1920x1080-4', '7-60-1920x1080']

for filename in files:
    print filename
    framePath = pathIn+filename+'/'
    frames = os.listdir(framePath)
    frames = sorted(frames)
    # get min and max frame number
    start = frames[0].split('.')[0].split('-')[0]
    end = frames[-1].split('.')[0].split('-')[0]

    outputFramePath = pathOut+filename+'/'
    if not os.path.exists(outputFramePath):
        os.makedirs(outputFramePath)
    # iterate all frame number and select the only one detected face for that frame
    res = startWithNumber(int(start), frames)
    res = sorted(res)
    # get target feature / histogram
    targetFeature = extract_features(framePath+res[0])
    targetHist = getHist(framePath+res[0])
    name = res[0].split('.')[0].split('-')[0]
    subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath+name+'.jpg', shell=True)
    for i in range(int(start)+1, int(end)+1):
        res = startWithNumber(i, frames)
        res = sorted(res)
        if len(res) == 0:
            continue
        elif len(res) == 1:
            tempFeature = extract_features(framePath+res[0])
            tempHist = getHist(framePath+res[0])
            if tempFeature is not None:
                d = cos_cdist(targetFeature, tempFeature)
            else:
                d = 0
            dh = compareHist(targetHist, tempHist)
            if (d > 0.5 and dh > 0.5):
                print res[0]
                continue
            else :
                if tempFeature is not None:
                    targetFeature = tempFeature
                targetHist = tempHist
                name = res[0].split('.')[0].split('-')[0]
                subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath+name+'.jpg', shell=True)
        else:
            # len(res) >1
            disDict = {}
            for imgName in res:
                tempFeature = extract_features(framePath + imgName)
                if tempFeature is None:
                    continue
                tempD = cos_cdist(targetFeature, tempFeature)
                disDict[str(tempD)] = imgName
            keys = sorted(disDict.keys())
            target = disDict[keys[0]]
            targetHist = getHist(framePath + target)
            targetFeature = extract_features(framePath + target)
            name = target.split('.')[0].split('-')[0]
            subprocess.call('cp ' + framePath+target + ' '+outputFramePath+name+'.jpg', shell=True)
