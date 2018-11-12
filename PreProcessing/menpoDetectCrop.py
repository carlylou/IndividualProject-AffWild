import glob, os

import numpy as np
import subprocess
import re
from menpo.visualize import print_progress


digits = re.compile(r'(\d+)')
_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def tokenize(filename):
    return tuple(int(token) if match else token
                 for token, match in
                 ((fragment, digits.search(fragment))
                  for fragment in digits.split(filename)))


import menpo.io as mio
from menpodetect.ffld2 import load_ffld2_frontal_face_detector

detector = load_ffld2_frontal_face_detector()




for utterance in print_progress(vids): # vids is a list with all videos such as 105_25_fsfas.mp4
  fps_check = subprocess.check_output('ffprobe '+whole_path_of_video+' 2>&1 | grep fps',shell=True)
  fps_check = str(fps_check, 'utf-8')
  fps = float(fps_check.split(' fps')[0].split(',')[-1][1:])  ## in our case we know fps=30 so you can just put this

  if not os.path.exists(path_where_you_want_the_uncropped_images_to_be_stored):
        os.makedirs(path_where_you_want_the_uncropped_images_to_be_stored)

  if not os.path.exists(same_as_before_with_cropped_images):
        os.makedirs(same_as_before_with_cropped_images)

  subprocess.call('ffmpeg -loglevel panic -i '+whole_path_of_video+' -vf fps='+ str(fps)+' '+path_where_you_want_the_uncropped_images_to_be_stored+'/%05d.jpg',shell=True)
  vv = mio.import_images(path_where_you_want_the_uncropped_images_to_be_stored+'/*.jpg')

  for cnt, im in enumerate(vv):
    name = '{0:05d}'.format(cnt+1)
    lns = detector(im)
    if im.landmarks.n_groups == 0:
        # there are no detections
        continue
    if im.landmarks.n_groups == 1:
      im.constrain_landmarks_to_bounds()
      mio.export_image(im.crop_to_landmarks(), cropped_image_location+'/'+name+'.jpg', extension=None, overwrite=True)
    elif  im.landmarks.n_groups > 1:
     for i in range(im.landmarks.n_groups):
       im.constrain_landmarks_to_bounds()
       mio.export_image(im.crop_to_landmarks(group='ffld2_'+str(i)), cropped_image_location+'/'+name+'_'+str(i)+'.jpg', extension=None, overwrite=True)
