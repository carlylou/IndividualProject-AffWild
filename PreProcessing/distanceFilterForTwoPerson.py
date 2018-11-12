#!/usr/bin/env python

import os
import cv2
import subprocess
import face_recognition

def startWithNumber(number, frameList):
    name = '{0:05d}'.format(number)
    result = filter(lambda x: x.startswith(name), frameList)
    return result

pathOut = ('/vol/bitbucket/ml9915/filterSimilarity/')
pathIn = ('/vol/bitbucket/ml9915/ffcropped3/')
files = ['69-25-854x480']

for filename in files:
    framePath = pathIn+filename+'/'
    frames = os.listdir(framePath)
    frames = sorted(frames)
    # get min and max frame number
    start = frames[0].split('.')[0].split('-')[0]
    end = frames[-1].split('.')[0].split('-')[0]

    outputFramePath1 = pathOut+filename+'-1/'
    if not os.path.exists(outputFramePath1):
        os.makedirs(outputFramePath1)
    outputFramePath2 = pathOut+filename+'-2/'
    if not os.path.exists(outputFramePath2):
        os.makedirs(outputFramePath2)
    # iterate all frame number and select the only one detected face for that frame
    res = startWithNumber(int(start), frames)
    res = sorted(res)
    # person1 = face_recognition.load_image_file(framePath+res[0])
    # print framePath+res[0]
    person2 = face_recognition.load_image_file(framePath+res[0])
    print framePath+res[0]
    # person1_encode = face_recognition.face_encodings(person1)[0]
    person2_encode = face_recognition.face_encodings(person2)[0]
    name = res[0].split('.')[0].split('-')[0]
    # subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath1+name+'.jpg', shell=True)
    subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath2+name+'.jpg', shell=True)

    for i in range(int(start)+1, int(end)+1):
        res = startWithNumber(i, frames)
        res = sorted(res)
        if len(res) == 0:
            continue
        if len(res) == 1:
            # img = face_recognition.load_image_file(framePath+res[0])
            # img_encode = face_recognition.face_encodings(img)
            # if len(img_encode) == 0:
            #     print res[0]
            #     continue
            # else:
                # distance1 = face_recognition.compare_faces([person1_encode], img_encode[0])
            name = res[0].split('.')[0].split('-')[0]
            subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath2+name+'.jpg', shell=True)
                # if distance1<distance2:
                #     subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath1+name+'.jpg', shell=True)
                # else:
                #     subprocess.call('cp ' + framePath+res[0] + ' '+outputFramePath2+name+'.jpg', shell=True)
        elif len(res)>1:
            # for person 1
            # disDict1 = {}
            # for imgName in res:
            #     img = face_recognition.load_image_file(framePath+imgName)
            #     img_encode = face_recognition.face_encodings(img)
            #     if len(img_encode) == 0:
            #         continue
            #     else:
            #         disDict1[str(img_encode[0])] = imgName
            # keys = sorted(disDict1.keys())
            # target = disDict1[keys[0]]
            # name = target.split('.')[0].split('-')[0]
            # subprocess.call('cp ' + framePath+target + ' '+outputFramePath1+name+'.jpg', shell=True)

            # for person 2
            disDict2 = {}
            for imgName in res:
                img = face_recognition.load_image_file(framePath+imgName)
                img_encode = face_recognition.face_encodings(img)
                if len(img_encode) == 0:
                    continue
                else:
                    disDict2[str(img_encode[0])] = imgName
            keys = sorted(disDict2.keys())
            if len(keys) == 0:
                print res[0] +' fail'
                continue
            target = disDict2[keys[0]]
            name = target.split('.')[0].split('-')[0]
            subprocess.call('cp ' + framePath+target + ' '+outputFramePath2+name+'.jpg', shell=True)
            # subprocess.call('cp ' + framePath+disDict2[keys[1]] + ' '+outputFramePath1+name+'.jpg', shell=True)
