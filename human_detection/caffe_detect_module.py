#!/usr/bin/env python
"""
detector.py is an out-of-the-box windowed detector
callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
Note that this model was trained for image classification and not detection,
and finetuning for detection can be expected to improve results.

The selective_search_ijcv_with_python code required for the selective search
proposal mode is available at
    https://github.com/sergeyk/selective_search_ijcv_with_python

TODO:
- batch up image filenames as well: don't want to load all of them into memory
- come up with a batching scheme that preserved order / keeps a unique ID
"""
import numpy as np
import pandas as pd
import os
import argparse
import time
import io

import caffe
import my_sliding_windows
import cv2
from imutils.object_detection import non_max_suppression
# import selective_search_ijcv_with_python as selective_search

caffe_detector = None

def init_caffe():

    global caffe_detector
    if caffe_detector is not None:
        return caffe_detector

    pycaffe_dir = os.path.dirname(__file__)

    #initialize all input variables
    mean, channel_swap = None, None
    context_pad = 16 #help="Amount of surrounding context to collect in input window."
    channel_swap = '2,1,0' #help="Order to permute input channels. The default converts " +
                            #"RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    raw_scale = 255.0 #help="Multiply raw input by this scale before preprocessing."
    input_scale = None #help="Multiply input features by this scale to finish preprocessing."
    mean_file = './data/ilsvrc12/imagenet_mean.binaryproto' #help="Data set image mean of H x W x K dimensions (numpy array). " +
                    #"Set to '' for no mean subtraction."
    pretrained_model = "./models/bvlc_reference_rcnn_ilsvrc13.caffemodel"  #help="Trained model weights file."
    model_def = "./models/deploy.prototxt" #help="Model definition file."
    labels_file = './models/det_synset_words.txt'

    if channel_swap:
        channel_swap = [int(s) for s in channel_swap.split(',')]
    # channel_swap = [int(s) for s in channel_swap.split(',')]

    caffe.set_mode_cpu()

    # Make detector.
    detector = caffe.Detector(model_def, pretrained_model, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap,
            context_pad=context_pad)

    caffe_detector = detector
    return detector

def get_caffe_detections(fname, img):

    detector = init_caffe()
    pretrained_model = "./models/bvlc_reference_rcnn_ilsvrc13.caffemodel"  #help="Trained model weights file."
    model_def = "./models/deploy.prototxt" #help="Model definition file."
    labels_file = './models/det_synset_words.txt'


    COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']
    
    boxes = my_sliding_windows.get_windows(fname, img, 40, 106) #50, 256
    TESTDATA = io.StringIO(my_sliding_windows.get_str_to_csv(boxes))

    # Load input.
    t = time.time()
    print("Loading input...")

    f = TESTDATA

    # Detect.
    # 123 index is human

    inputs = pd.read_csv(f, sep=',', dtype={'filename': str})
    inputs.set_index('filename', inplace=True)

    # Unpack sequence of (image filename, windows).
    images_windows = [
        (ix, inputs.iloc[np.where(inputs.index == ix)][COORD_COLS].values)
        for ix in inputs.index.unique()
    ]
    detections = detector.detect_windows(images_windows)

    #using selective search
    # detections = detector.detect_selective_search(inputs)

    print("Processed {} windows in {:.3f} s.".format(len(detections),
                                                     time.time() - t))

    #loop through the output and filter humans

    #get labels for classes
    with open(labels_file) as f:
        labels_df = [
            {
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            }
            for l in f.readlines()
        ]

    # print "detections object: ", detections[0]

    # imgcpy = cv2.imread(fname)
    totalrects = []

    for i in detections:

        prdt = i['prediction']
        maxp = max(prdt)
        prdt = map(lambda x: (x), prdt);
        maxi = prdt.index(maxp)

        if maxi == 123 and maxp > 0.5:
            # print "human detected!"
            coord = i['window']
            totalrects.append([coord[1], coord[0], coord[3], coord[2]])

    print "Total humans detected: ", str(len(totalrects))
    #non max suppression
    totalrects = np.array(totalrects)
    processedboxes = non_max_suppression(totalrects, probs=None, overlapThresh=0.55) #overlapThresh default 0.65
    print "Total humans after NMS correction: ", str(len(processedboxes))

    return processedboxes
