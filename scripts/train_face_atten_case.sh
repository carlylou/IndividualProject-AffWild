#!/bin/bash
#PBS -l nodes=gpu05

echo Start training:
. /vol/gpudata/cuda/8.0.61-cudnn.7.0.2/setup.sh
source /vol/gpudata/ml9915/Anaconda/bin/activate gpu-2.7
/vol/gpudata/ml9915/IndividualProject/train_face_atten_case2.py
echo End training:
