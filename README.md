# IndividualProject-AffWild
Individual Project in Imperial College London


All train/evaluate/test scripts are stored in scripts folder and their original path should be in current path. If they need to be run, they should be moved to current directory.


All python code is written under python 2.7.


All train, validate and test scripts run in environment python 2.7 and cuda version is 8.0.61-cudnn.7.0.2.
Every time run the script code, the following two lines command will be executed on GPU cluster:
. /vol/gpudata/cuda/8.0.61-cudnn.7.0.2/setup.sh
source /vol/gpudata/ml9915/Anaconda/bin/activate gpu-2.7

Exception for IndRNN model, IndRNN model need python 3.4 and cuda version 9.0.176:
source /vol/gpudata/ml9915/Anaconda/bin/activate gpu-3.4
. /vol/gpudata/cuda/9.0.176/setup.sh


All best performing check points for each architecture are recorded in :
archive/comparison.txt



Sample run:
For testing best performing model on valence which has 0.555 CCC value on valence:
archive/test_best_valence_performance.py

For testing best performing model on arousal which has 0.499 CCC value on arousal:
archive/test_best_arousal_performance.py





In the following, the details of where the code is for each step:

1. downloading video code:
archive/otherTool/youtube-dl.ipynb
archive/otherTool/DownloadFromYoutube.ipynb

2. converting to MP4: using Format Factory as stated in Individual Report

3. trim videos: using iMovie and FFmpeg: FFmpeg is executed in terminal with command line :
ffmpeg -ss start_time -i input_video -t duration -vcodec copy -acodec copy output_video

4. converting to 30FPS:
ffmpeg -i input_video -r 30 -c:v libx264 output_video

5. face detection:
archive/PreProcessing/ffmenpoDC2.py

6. filter cropped detected objects:
archive/PreProcessing/filterCropped1.py
archive/PreProcessing/faceSimilarity.py
archive/PreProcessing/faceSimilarityForTwo.py


7. matching process:

archive/otherTool/getAnnotationNNPerFrame.ipynb
archive/otherTool/annotationDistributionHistogram.ipynb
archive/otherTool/histogramVACropped.ipynb
archive/otherTool/croppedFramesAnnotationHist.ipynb


8. write data into TFRecords:
archive/serializeData/loadTrainData.py
archive/serializeData/loadTestData.py
archive/serializeData/loadValidationData.py


9. import data into neural network:

archive/data_utils_mean.py
10. CNN model:

ResNet50: archive/ResNet/*.py
VGGFace: archive/vgg_face.py
DenseNet: archive/CNNs/densenet.py


11. all RNN models:
archive/models.py

12. CCC metric and loss:

archive/metrics.py
archive/losses.py

13. train and evaluation scripts:
all python files in archive are used for train / validate / test with respective model name


14. for final test:
archive/test_all.py

15. for ploting the prediction and ground truth:
archive/plot_pred_label.py
