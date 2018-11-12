#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

FRAMES_VALID = 135145
def getAnnotationsList(fileName, directory):
    pairs = []
    file = open(directory+fileName, 'r')
    for line in file:
        pairs.append(line.split())
    pairs = np.array(pairs, dtype=np.float)
    return pairs

DATA_PATH = '/vol/gpudata/ml9915/summary/face_atten_case3_70/test/'
def main():
    # arousal , valence
    name = 'arousal'
    filenName  = name + '_test.txt'
    pairs = getAnnotationsList(filenName, DATA_PATH)
    length = np.shape(pairs)[0]
    start = 90000
    end = 120000

    # x = np.arange(0, length)
    # print length
    # plt.plot(x, pairs[:, 0])
    # plt.plot(x, pairs[:, 1])

    x = np.arange(start, end)
    print length
    plt.plot(x, pairs[start:end, 0])
    plt.plot(x, pairs[start:end, 1])

    plt.legend(['predictions', 'ground_truth'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
