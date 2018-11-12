#!/usr/bin/env python

import os
import numpy as np
from menpo.visualize import print_progress
import menpo
import menpo.io as mio
from menpodetect.ffld2 import load_ffld2_frontal_face_detector
from os.path import join

detector = load_ffld2_frontal_face_detector()

pb = ('/vol/bitbucket/ml9915/annotateVideos1/')
pout = ('/vol/bitbucket/ml9915/cropped1/')
files = os.listdir(pb)
files = filter(lambda x: not x.startswith("._"), files)

for filename in print_progress(files):
    if(filename == '.DS_Store'):
        continue
    if not os.path.exists(pout + filename.split('.')[0] + '/'):
        os.makedirs(pout + filename.split('.')[0] + '/')
    else:
        continue
    # import the video
    ims = mio.import_video(join(pb, filename))
    for cnt, im in enumerate(ims):
        lns = detector(im)
        name = '{0:05d}'.format(cnt+1)
        if im.landmarks.n_groups == 0:
            continue
        if im.landmarks.n_groups == 1:
            im.constrain_landmarks_to_bounds()
            mio.export_image(im.crop_to_landmarks(), pout + filename.split('.')[0] + '/'+name+'.jpg', extension=None, overwrite=True)
        elif im.landmarks.n_groups > 1:
            groupNo = 0
            for key in im.landmarks.keys():
                groupNo = groupNo+1
                im.constrain_landmarks_to_bounds()
                mio.export_image(im.crop_to_landmarks(group=key), pout + filename.split('.')[0] +'/'+name+'_'+str(groupNo)+'.jpg', extension=None, overwrite=True)
