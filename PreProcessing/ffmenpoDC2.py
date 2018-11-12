#!/usr/bin/env python

import os
import numpy as np
from menpo.visualize import print_progress
import menpo
import menpo.io as mio
from menpodetect.ffld2 import load_ffld2_frontal_face_detector
from os.path import join
import subprocess

detector = load_ffld2_frontal_face_detector()

pb = ('/vol/bitbucket/ml9915/annotateVideos2/')
pcrop = ('/vol/bitbucket/ml9915/ffcropped2/')
puncrop = ('/vol/bitbucket/ml9915/ffuncropped2/')

files = os.listdir(pb)
files = filter(lambda x: not x.startswith("._"), files)

for filename in print_progress(files):
    if(filename == '.DS_Store'):
        continue
    if not os.path.exists(pcrop + filename.split('.')[0] + '/'):
        os.makedirs(pcrop + filename.split('.')[0] + '/')
    if not os.path.exists(puncrop + filename.split('.')[0] + '/'):
        os.makedirs(puncrop + filename.split('.')[0] + '/')

    fps = 30
    subprocess.call('ffmpeg -loglevel panic -i '+ pb+filename +' -vf fps='+ str(fps)+' '+ puncrop+filename.split('.')[0] +'/%05d.jpg',shell=True)
    vv = mio.import_images(puncrop+filename.split('.')[0]+'/*.jpg')

    # import the video
    for cnt, im in enumerate(vv):
        lns = detector(im)
        name = '{0:05d}'.format(cnt+1)
        if im.landmarks.n_groups == 0:
            continue
        if im.landmarks.n_groups == 1:
            im.constrain_landmarks_to_bounds()
            mio.export_image(im.crop_to_landmarks(), pcrop + filename.split('.')[0] + '/'+name+'.jpg', extension=None, overwrite=True)
        elif im.landmarks.n_groups > 1:
            keys = im.landmarks.keys()
            keys = sorted(keys)
            for key in keys:
                im.constrain_landmarks_to_bounds()
                mio.export_image(im.crop_to_landmarks(group=key), pcrop + filename.split('.')[0] +'/'+name+'-'+key+'.jpg', extension=None, overwrite=True)
