#!/usr/bin/env python

import os
import cv2

# get all detected faces start with a particular frame number
def startWithNumber(number, frameList):
    name = '{0:05d}'.format(number)
    result = filter(lambda x: x.startswith(name), frameList)
    return result


pathIn = ('/vol/bitbucket/ml9915/ffcropped1/')
pathOut = ('/vol/bitbucket/ml9915/filtercropped1/')

files = os.listdir(pathIn)
files = filter(lambda x: not x.startswith("._"), files)

# iterate all video name
for filename in files:
    if(filename == '.DS_Store'):
        continue
    # get all frames jpg file in this directory
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
    for i in range(int(start), int(end)+1):
        res = startWithNumber(i, frames)
        if len(res) == 0:
            continue
        elif len(res) == 1:
            img = cv2.imread(framePath+res[0])
            cv2.imwrite(outputFramePath+res[0], img)
        elif len(res) >1 :
            res = sorted(res)
            name = '{0:05d}'.format(i)
            img = cv2.imread(framePath+res[0])
            cv2.imwrite(outputFramePath+name+'.jpg', img)
