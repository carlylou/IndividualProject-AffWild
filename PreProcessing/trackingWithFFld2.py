#!/usr/bin/env python

import os
import glob
import cv2
import dlib
import numpy as np

import menpo
from os.path import isdir, join
import menpo.io as mio
from menpodetect.ffld2 import load_ffld2_frontal_face_detector

puncrop = ('/vol/bitbucket/ml9915/ffuncropped1/')
pout = ('/vol/bitbucket/ml9915/tracking/')
files = ['30-30-1920x1080']

KCFtracker1 = cv2.TrackerKCF_create()
KCFtracker2 = cv2.TrackerKCF_create()

for filename in files:
    for k, f in enumerate(sorted(glob.glob(os.path.join(puncrop+filename, "*.jpg")))):
        img = cv2.imread(f)
        name = f.split('/')[-1] # like 00023.jpg
        # We need to initialize the tracker on the first frame
        if k == 0:
            detector = load_ffld2_frontal_face_detector()
            im = mio.import_image(f)
            lns = detector(im)
            keys = im.landmarks.keys()
            points1 = im.landmarks[keys[0]].bounding_box().points
            points2 = im.landmarks[keys[1]].bounding_box().points
            # tracker 1
            x1, y1, x2, y2 = int(points1[0][0]), int(points1[0][1]), int(points1[2][0]), int(points1[2][1])
            KCFtracker1.init(img, (x1, y1, x2-x1, y2-y1))
            crop_img1 = img[y1:y2, x1:x2]
            pout1 = pout+filename+'-tracker1/'
            if not os.path.exists(pout1):
                os.makedirs(pout1)
            cv2.imwrite(pout1+name, crop_img1)
            # tracker 2
            x1, y1, x2, y2 = int(points2[0][0]), int(points2[0][1]), int(points2[2][0]), int(points2[2][1])
            KCFtracker2.init(img, (x1, y1, x2-x1, y2-y1))
            crop_img2 = img[y1:y2, x1:x2]
            pout2 = pout+filename+'-tracker2/'
            if not os.path.exists(pout2):
                os.makedirs(pout2)
            cv2.imwrite(pout2+name, crop_img2)
        else:
            # KCFtracker1 Green
            ok, KCFrect1 = KCFtracker1.update(img)
            ok, KCFrect2 = KCFtracker2.update(img)
            KCF1_P1 = (int(KCFrect1[0]), int(KCFrect1[1]))
            KCF1_P2 = (int(KCFrect1[0]+KCFrect1[2]), int(KCFrect1[1]+KCFrect1[3]))
            KCF2_P1 = (int(KCFrect2[0]), int(KCFrect2[1]))
            KCF2_P2 = (int(KCFrect2[0]+KCFrect2[2]), int(KCFrect2[1]+KCFrect2[3]))
            # Tracker 1 combine:
            x1 = KCF1_P1[0]
            y1 = KCF1_P1[1]
            x2 = KCF1_P2[0]
            y2 = KCF1_P2[1]
            crop_img1 = img[y1:y2, x1:x2]
            # Tracker 2 combine:
            x1 = KCF2_P1[0]
            y1 = KCF2_P1[1]
            x2 = KCF2_P2[0]
            y2 = KCF2_P2[1]
            crop_img2 = img[y1:y2, x1:x2]
            cv2.imwrite(pout1+name, crop_img1)
            cv2.imwrite(pout2+name, crop_img2)
