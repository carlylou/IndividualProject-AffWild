#!/usr/bin/env python

import menpo
from os.path import isdir, join
import numpy as np
import menpo.io as mio
from menpodetect.ffld2 import load_ffld2_frontal_face_detector
import os

# process all frames iteratively.
def storeBB(ims, filename):
    p_out = '/vol/bitbucket/ml9915/landmarksLeft/'
    bboxes = {}
    for cnt, im in enumerate(ims):
        if im.n_channels == 3:
            im = im.as_greyscale()
    #   ffld2 detector returns bounding_boxes
        lns = ffld2_detector(im)
        if im.landmarks.n_groups == 0:
            # there are no detections
            print 'error'
            continue
        name = '{0:06d}'.format(cnt)
        bboxes[name] = lns
    # export the boundingbox
    mio.export_pickle(bboxes, p_out + filename +'.pkl', overwrite=True)

# decide which detector to use:
ffld2_detector = load_ffld2_frontal_face_detector()
# input path
pbLeft = ('/vol/bitbucket/ml9915/annotateVideosLeft/')
filesLeft = os.listdir(pbLeft)
filesLeft = filter(lambda x: not x.startswith("._"), filesLeft)

for filename in filesLeft:
    if(filename == '.DS_Store'):
        continue
    print filename
    # import the video
    ims = mio.import_video(join(pbLeft, filename))
    storeBB(ims, filename.split('.')[0])
