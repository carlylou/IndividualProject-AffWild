#!/usr/bin/env python

# use jpg file end with ffld2_n replace jpg file but keep the filename

import os
import subprocess


def startWithNumber(number, frameList):
    name = '{0:05d}'.format(number)
    result = filter(lambda x: x.startswith(name), frameList)
    return result

cleanPath = ('/vol/bitbucket/ml9915/filtercropped1/')
needCleanFile = ['6-30-1920x1080-p1']

for filename in needCleanFile:
    framePath = cleanPath+filename+'/'
    frames = os.listdir(framePath)
    drop = filter(lambda x: x.startswith('.'), frames)
    for item in drop:
        frames.remove(item)
    frames = sorted(frames)
    # get min and max frame number
    start = frames[0].split('.')[0].split('-')[0]
    print frames[0]
    print start
    end = frames[-1].split('.')[0].split('-')[0]
    print end
    for i in range(int(start), int(end)+1):
        name = '{0:05d}'.format(i)
        res = startWithNumber(i, frames)
        if len(res) <1:
            continue
        real = res[0].split('.')[0]
        if name == real:
            continue
        else:
            subprocess.call('mv ' + framePath+res[0] + ' '+framePath+name+'.jpg', shell=True)
        # if len(res) <= 1:
        #     continue
        # else:
        #     res = sorted(res)
        #     subprocess.call('rm ' + framePath+res[0], shell=True)
        #     subprocess.call('mv ' + framePath+res[-1] + ' '+framePath+res[0], shell=True)
