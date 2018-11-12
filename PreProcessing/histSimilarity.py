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

def startWithNumber(number, frameList):
    name = '{0:05d}'.format(number)
    result = filter(lambda x: x.startswith(name), frameList)
    return result

pathOut = ('/vol/bitbucket/ml9915/filterSimilarity/')
pathIn = ('/vol/bitbucket/ml9915/ffcropped4/')
files = ['139-14-720x480']

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
    targetHist = getHist(framePath+res[0])
    name = res[0].split('.')[0].split('-')[0]
    subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath+name+'.jpg', shell=True)
    for i in range(int(start)+1, int(end)+1):
        res = startWithNumber(i, frames)
        res = sorted(res)
        if len(res) == 0:
            continue
        elif len(res) == 1:
            tempHist = getHist(framePath+res[0])
            dh = compareHist(targetHist, tempHist)
            if (dh > 0.5):
                print res[0]
                continue
            else :
                targetHist = tempHist
                name = res[0].split('.')[0].split('-')[0]
                subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath+name+'.jpg', shell=True)
        else:
            # len(res) >1
            disDict = {}
            for imgName in res:
                tempHist = getHist(framePath + imgName)
                tempD = compareHist(targetHist, tempHist)
                disDict[str(tempD)] = imgName
            keys = sorted(disDict.keys())
            target = disDict[keys[0]]
            targetHist = getHist(framePath + target)
            name = target.split('.')[0].split('-')[0]
            subprocess.call('cp ' + framePath+target + ' '+outputFramePath+name+'.jpg', shell=True)
