webcamDetector
==============

Use a Webcam with OpenCV and TensorFlow to detect objects

Records output video to the "output" directory

To use:

1. Install TensorFlow Object Detection: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md


2. Test TensorFlow and OpenCV:

    import sys
    sys.version
Outputs: '3.6.2 (default, Jul 17 2017, 16:44:32) \n[GCC 4.2.1 Compatible Apple LLVM 7.0.2 (clang-700.1.81)]'
    
    import tensorflow as tf
    tf.__version__
Outputs: '1.3.0'

    import cv2
    cv2.__version__
Outputs: '3.3.0'


3. download and run:

    git clone https://github.com/alkari/webcamDetector.git
    cd webcamDetector
    python3 webcamDetector.py