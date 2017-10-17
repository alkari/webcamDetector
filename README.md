webcamDetector
==============

Use a Webcam with OpenCV and TensorFlow to detect objects


To use:

1. Upgrade Python

```
$ brew install --upgrade python3
$ brew link python3
$ python -V
```
Outputs: 'Python 3.6.2'


2. Install TensorFlow from https://www.tensorflow.org/install/

Test TensorFlow in python:
 
```python
>>> import tensorflow as tf
>>> tf.__version__
```
Outputs: '1.3.0'

3. Install TensorFlow Object Detection: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

Remember to run this from tensorflow/models/object_detection: 
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

4. Intall OpenCV

The following should work on MacOS (from https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/):
```
$ brew tap homebrew/science
$ brew update
$ brew install data_utils
$ brew install ffmpeg
$ brew install opencv3 --with-python3 --with-ffmpeg -v
```
Test in python:

```python
>>> import cv2
>>> cv2.__version__
```
Outputs: '3.3.0'


5. download and run:

```
$ git clone https://github.com/alkari/webcamDetector.git
$ cd webcamDetector
$ python3 webcamDetector.py
```

The script will download and unpack pretrained model used for detection. It might take a little while depending on your network speed.

5. Press 'Q' on your keyboard to exit.

## Recorded detection videos are stored in the "output" directory.

