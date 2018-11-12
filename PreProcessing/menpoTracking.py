# venv-2.7 menpo detector plus opencv tracking


import os
import glob
import cv2
import dlib
import numpy as np


KCFtracker1 = cv2.TrackerKCF_create()
KCFtracker2 = cv2.TrackerKCF_create()

for k, f in enumerate(sorted(glob.glob(os.path.join(video_folder, "*.jpg")))):
    img = cv2.imread(f)
    # We need to initialize the tracker on the first frame
    if k == 0:
        detector = menpoFFld2.get_frontal_face_detector()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_rects = detector(gray, 0)
        d1 = face_rects[0]
        d2 = face_rects[1]
        
        x1, y1, x2, y2, w, h = d1.left(), d1.top(), d1.right() + 1, d1.bottom() + 1, d1.width(), d1.height()
        KCFtracker1.init(img, (x1, y1, w, h))
        CSRTtracker1.init(img, (x1, y1, w, h))
        crop_img1 = img[y1:y2, x1:x2]
        name = f.split('/')[-1].split('.')[0]
        pout1 = '../cropped/30-30-1920x1080'+'-tracker1-'+str(k/1000)+'/'
        if not os.path.exists(pout1):
            os.makedirs(pout1)
        cv2.imwrite(pout1+name+'.jpg', crop_img1)
        
        x1, y1, x2, y2, w, h = d2.left(), d2.top(), d2.right() + 1, d2.bottom() + 1, d2.width(), d2.height()
        KCFtracker2.init(img, (x1, y1, w, h))
        CSRTtracker2.init(img, (x1, y1, w, h))
        crop_img2 = img[y1:y2, x1:x2]
        name = f.split('/')[-1].split('.')[0]
        pout2 = '../cropped/30-30-1920x1080'+'-tracker2-'+str(k/1000)+'/'
        if not os.path.exists(pout2):
            os.makedirs(pout2)
        cv2.imwrite(pout2+name+'.jpg', crop_img2)
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
        
        name = f.split('/')[-1].split('.')[0]
        cv2.imwrite(pout1+name+'.jpg',crop_img1)
        cv2.imwrite(pout2+name+'.jpg',crop_img2)