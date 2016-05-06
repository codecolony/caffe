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

# CROP_MODES = ['list', 'selective_search']
COORD_COLS = ['ymin', 'xmin', 'ymax', 'xmax']


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)
    # pycaffe_dir = os.path.dirname("~/installations/caffe/python")

    parser = argparse.ArgumentParser()
    # Required arguments: input and output.
    # parser.add_argument(
    #     "input_file",
    #     help="Input txt/csv filename. If .txt, must be list of filenames.\
    #     If .csv, must be comma-separated file with header\
    #     'filename, xmin, ymin, xmax, ymax'"
    # )
    # parser.add_argument(
    #     "output_file",
    #     help="Output h5/csv filename. Format depends on extension."
    # )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        # default=os.path.join(pycaffe_dir,
        #         "./models/bvlc_reference_caffenet/deploy.prototxt"),
        # default="./vgg16/VGG_ILSVRC_16_layers_deploy.prototxt",
        # default="./googleNet/deploy.prototxt",
        default="./models/deploy.prototxt",
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default="./models/bvlc_reference_rcnn_ilsvrc13.caffemodel",
        # default=os.path.join(pycaffe_dir,
        #         "./models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        # default="./vgg16/VGG_ILSVRC_16_layers.caffemodel",
        # default="./googleNet/bvlc_googlenet.caffemodel",
        

        help="Trained model weights file."
    )
    # parser.add_argument(
    #     "--crop_mode",
    #     default="selective_search",
    #     choices=CROP_MODES,
    #     help="How to generate windows for detection."
    # )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of H x W x K dimensions (numpy array). " +
             "Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."

    )
    parser.add_argument(
        "--context_pad",
        type=int,
        default='16',
        help="Amount of surrounding context to collect in input window."
    )
    args = parser.parse_args()

    mean, channel_swap = None, None
    # mean = np.load(args.mean_file)
    # if mean.shape[1:] != (1, 1):
    #     mean = mean.mean(1).mean(1)

    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    caffe.set_mode_cpu()

    # Make detector.
    detector = caffe.Detector(args.model_def, args.pretrained_model, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap,
            context_pad=args.context_pad)

#     TESTDATA=io.StringIO(u"""filename,xmin,ymin,xmax,ymax
# group.jpg,42,39,107,317
# group.jpg,62,69,127,327
# """)

    fname = "group.jpg"
    image = cv2.imread("group.jpg")
    boxes = my_sliding_windows.get_windows(fname, image, 50, 256) #50, 256
    TESTDATA = io.StringIO(my_sliding_windows.get_str_to_csv(boxes))

    # Load input.
    t = time.time()
    print("Loading input...")

    f = TESTDATA
    
    # inputs = [_.strip() for _ in f]
    # raise Exception("Unknown input file type: not in txt or csv.")

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
    with open('./models/det_synset_words.txt') as f:
        labels_df = [
            {
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            }
            for l in f.readlines()
        ]

    # print "detections object: ", detections[0]

    imgcpy = cv2.imread(fname)
    totalrects = []

    for i in detections:

        prdt = i['prediction']
        maxp = max(prdt)
        prdt = map(lambda x: (x), prdt);
        maxi = prdt.index(maxp)

        print "maxp: ", maxp
        if maxi == 123 and maxp > 0.7:
            # print "human detected!"
            coord = i['window']
            # print coord
            # cv2.rectangle(imgcpy, (coord[1], coord[0]), (coord[3], coord[2]), (0, 255, 0), 2)
            totalrects.append([coord[1], coord[0], coord[3], coord[2]])

    print "Total humans detected: ", str(len(totalrects))
    #non max suppression
    totalrects = np.array(totalrects)
    processedboxes = non_max_suppression(totalrects, probs=None, overlapThresh=0.55) #overlapThresh default 0.65
    print "Total humans after NMS correction: ", str(len(processedboxes))

    for i in processedboxes:
        # print "human detected after NMS!"
        cv2.rectangle(imgcpy, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)

    cv2.imshow('labelled_image',imgcpy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print "detections: ", detections[1]
    # prdt = detections[1]['prediction']

    # print "prdt: ", prdt

    # indices = (-prdt).argsort()[:5]
    # print "indices: ", indices
    

    # maxp = max(prdt)
    # prdt = map(lambda x: (x), prdt);
    # maxi = prdt.index(maxp)
    # print "index of max: ", maxi
    # print "max prediction: ", maxp

    

    # print "labels: ", labels_df
    # print "class label of max prediction: ", labels_df.index(maxp)
    # print "Length of labels array: ", len(labels_df)
    # print "class : ", labels_df[maxi]

    # predictions = labels_df[indices[0]]
    # print "prd: ", predictions
    # predictions = labels_df[indices[1]]
    # print "prd: ", predictions
    # predictions = labels_df[indices[2]]
    # print "prd: ", predictions
    # predictions = labels_df[indices[3]]
    # print "prd: ", predictions
    # predictions = labels_df[indices[4]]
    # print "prd: ", predictions


if __name__ == "__main__":
    import sys
    main(sys.argv)



# In Python 3:

# >>> import io
# >>> import csv
# >>> output = io.StringIO()
# >>> csvdata = [1,2,'a','He said "what do you mean?"',"Whoa!\nNewlines!"]
# >>> writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
# >>> writer.writerow(csvdata)
# 59
# >>> output.getvalue()
# '1,2,"a","He said ""what do you mean?""","Whoa!\nNewlines!"\r\n'

# Some details need to be changed a bit for Python 2:

# >>> output = io.BytesIO()
# >>> writer = csv.writer(output)
# >>> writer.writerow(csvdata)
# 57L
# >>> output.getvalue()
# '1,2,a,"He said ""what do you mean?""","Whoa!\nNewlines!"\r\n'