#!/usr/bin/python

import cv2
import numpy as np
import sys
import os.path
import caffe
import os
from PIL import Image
import math
# import textsearch

#global variables for image width and height
img_w = 0
img_h = 0
word_gap = 0.2

net = None
transformer = None
imagenet_labels_filename = None

os_type = "linux"

if len(sys.argv) != 3:
    print "%s input_file output_file" % (sys.argv[0])
    sys.exit()
else:
    input_file = sys.argv[1]
    output_file = sys.argv[2]

if not os.path.isfile(input_file):
    print "No such file '%s'" % input_file
    sys.exit()

DEBUG = 0


# Determine pixel intensity
# Apparently human eyes register colors differently.
# TVs use this formula to determine
# pixel intensity = 0.30R + 0.59G + 0.11B
def ii(xx, yy):
    global img, img_y, img_x
    if yy >= img_y or xx >= img_x:
        #print "pixel out of bounds ("+str(y)+","+str(x)+")"
        return 0
    pixel = img[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]


# A quick test to check whether the contour is
# a connected shape
def connected(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1


# Helper function to return a given contour
def c(index):
    global contours
    return contours[index]


# Count the number of real children
def count_children(index, h_, contour):
    # No children
    if h_[index][2] < 0:
        return 0
    else:
        #If the first child is a contour we care about
        # then count it, otherwise don't
        if keep(c(h_[index][2])):
            count = 1
        else:
            count = 0

            # Also count all of the child's siblings and their children
        count += count_siblings(h_[index][2], h_, contour, True)
        return count


# Quick check to test if the contour is a child
def is_child(index, h_):
    return get_parent(index, h_) > 0


# Get the first parent of the contour that we care about
def get_parent(index, h_):
    parent = h_[index][3]
    while not keep(c(parent)) and parent > 0:
        parent = h_[parent][3]

    return parent


# Count the number of relevant siblings of a contour
def count_siblings(index, h_, contour, inc_children=False):
    # Include the children if necessary
    if inc_children:
        count = count_children(index, h_, contour)
    else:
        count = 0

    # Look ahead
    p_ = h_[index][0]
    while p_ > 0:
        if keep(c(p_)):
            count += 1
        if inc_children:
            count += count_children(p_, h_, contour)
        p_ = h_[p_][0]

    # Look behind
    n = h_[index][1]
    while n > 0:
        if keep(c(n)):
            count += 1
        if inc_children:
            count += count_children(n, h_, contour)
        n = h_[n][1]
    return count


# Whether we care about this contour
def keep(contour):
    return keep_box(contour) and connected(contour)


# Whether we should keep the containing box of this
# contour based on it's shape
def keep_box(contour):
    xx, yy, w_, h_ = cv2.boundingRect(contour)

    # width and height need to be floats
    w_ *= 1.0
    h_ *= 1.0

    # Test it's shape - if it's too oblong or tall it's
    # probably not a real character
    if w_ / h_ < 0.1 or w_ / h_ > 5 or w_ / img_w > 0.6 or h_ / img_h > 0.6: #check width and height wrt total image width and height and reject or accept accordingly
        if DEBUG:
            print "\t Rejected because of shape: (" + str(xx) + "," + str(yy) + "," + str(w_) + "," + str(h_) + ")" + \
                  str(w_ / h_)
        return False
    
    # check size of the box
    if ((w_ * h_) > ((img_x * img_y) / 5)) or ((w_ * h_) < 15):
        if DEBUG:
            print "\t Rejected because of size"
        return False

    return True


def include_box(index, h_, contour):
    if DEBUG:
        print str(index) + ":"
        if is_child(index, h_):
            print "\tIs a child"
            print "\tparent " + str(get_parent(index, h_)) + " has " + str(
                count_children(get_parent(index, h_), h_, contour)) + " children"
            print "\thas " + str(count_children(index, h_, contour)) + " children"

    if is_child(index, h_) and count_children(get_parent(index, h_), h_, contour) <= 2:
        if DEBUG:
            print "\t skipping: is an interior to a letter"
        return False

    if count_children(index, h_, contour) > 2:
        if DEBUG:
            print "\t skipping, is a container of letters"
        return False

    if DEBUG:
        print "\t keeping"
    return True

def init_caffe_model():
    if os_type=="osx":
        # Make sure that caffe is on the python path:
        caffe_root = '<caffe root>'  # this file is expected to be in {caffe_root}/examples
        sys.path.insert(0, caffe_root + 'python')

        if not os.path.isfile(caffe_root + 'examples/alphabet/char74k/char74k_iter_10000.caffemodel'):
            print("cannot find pre-trained CaffeNet model...")
            sys.exit(0)

        caffe.set_mode_cpu()
        net = caffe.Net(caffe_root + 'examples/alphabet/char74k/lenet.prototxt',
                        caffe_root + 'examples/alphabet/char74k/char74k_iter_10000.caffemodel',
                        caffe.TEST)

        imagenet_labels_filename = caffe_root + 'examples/alphabet/char74k/synset_words.txt'
    else:
        # Make sure that caffe is on the python path:
        caffe_root = '<model root>'  # this file is expected to be in {caffe_root}/examples
        sys.path.insert(0, caffe_root + 'python')

        if not os.path.isfile(caffe_root + 'char74k_iter_10000.caffemodel'):
            print("cannot find pre-trained CaffeNet model...")
            sys.exit(0)

        caffe.set_mode_cpu()
        net = caffe.Net(caffe_root + 'lenet.prototxt',
                        caffe_root + 'char74k_iter_10000.caffemodel',
                        caffe.TEST)

        imagenet_labels_filename = caffe_root + 'synset_words.txt'

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    # transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    return net, transformer, imagenet_labels_filename

def net_process_char(input_image):

    # if os_type=="osx":
    #     # Make sure that caffe is on the python path:
    #     caffe_root = '<caffe root>'  # this file is expected to be in {caffe_root}/examples
    #     sys.path.insert(0, caffe_root + 'python')

    #     if not os.path.isfile(caffe_root + 'examples/alphabet/char74k/char74k_iter_10000.caffemodel'):
    #         print("cannot find pre-trained CaffeNet model...")
    #         sys.exit(0)

    #     caffe.set_mode_cpu()
    #     net = caffe.Net(caffe_root + 'examples/alphabet/char74k/lenet.prototxt',
    #                     caffe_root + 'examples/alphabet/char74k/char74k_iter_10000.caffemodel',
    #                     caffe.TEST)

    #     imagenet_labels_filename = caffe_root + 'examples/alphabet/char74k/synset_words.txt'
    # else:
    #     # Make sure that caffe is on the python path:
    #     caffe_root = '<model root>'  # this file is expected to be in {caffe_root}/examples
    #     sys.path.insert(0, caffe_root + 'python')

    #     if not os.path.isfile(caffe_root + 'char74k_iter_10000.caffemodel'):
    #         print("cannot find pre-trained CaffeNet model...")
    #         sys.exit(0)

    #     caffe.set_mode_cpu()
    #     net = caffe.Net(caffe_root + 'lenet.prototxt',
    #                     caffe_root + 'char74k_iter_10000.caffemodel',
    #                     caffe.TEST)

    #     imagenet_labels_filename = caffe_root + 'synset_words.txt'

    # # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_transpose('data', (2,0,1))
    # # transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    # transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    # transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    global net, transformer, imagenet_labels_filename

    if(net == None):
        net, transformer, imagenet_labels_filename = init_caffe_model()

    # set net to batch size of 64
    net.blobs['data'].reshape(1,3,28,28)

    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(input_image))
    out = net.forward()
    # print("Predicted class is #{}.".format(out['prob'].argmax()))
    # print "out: ", out['prob']

    # load labels
    # imagenet_labels_filename = caffe_root + 'examples/alphabet/char74k/synset_words.txt'
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        print 'synset_words.txt cannot be found'

    # sort top k predictions from softmax output
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    # print labels[top_k]
    return labels[top_k][0]

def classify_char(input_image):

    if os_type=="osx":
        # Make sure that caffe is on the python path:
        caffe_root = '<caffe root>'  # this file is expected to be in {caffe_root}/examples
        sys.path.insert(0, caffe_root + 'python')

        if not os.path.isfile(caffe_root + 'examples/alphabet/char74k/char74k_iter_10000.caffemodel'):
            print("cannot find pre-trained CaffeNet model...")
            sys.exit(0)

        caffe.set_mode_cpu()

        ####### Try classifier way ###########
        # Make classifier.
        classifier = caffe.Classifier(caffe_root + 'examples/alphabet/char74k/lenet.prototxt', 
                caffe_root + 'examples/alphabet/char74k/char74k_iter_10000.caffemodel',
                image_dims=(28,28), mean=None,
                input_scale=None, raw_scale=255.0,
                channel_swap=(2,1,0))

        imagenet_labels_filename = caffe_root + 'examples/alphabet/char74k/synset_words.txt'
    else:
        # Make sure that caffe is on the python path:
        caffe_root = '<model root>'  # this file is expected to be in {caffe_root}/examples
        sys.path.insert(0, caffe_root + 'python')

        if not os.path.isfile(caffe_root + 'char74k_iter_10000.caffemodel'):
            print("cannot find pre-trained CaffeNet model...")
            sys.exit(0)

        caffe.set_mode_cpu()

        ####### Try classifier way ###########
        # Make classifier.
        classifier = caffe.Classifier(caffe_root + 'lenet.prototxt', 
                caffe_root + 'char74k_iter_10000.caffemodel',
                image_dims=(28,28), mean=None,
                input_scale=None, raw_scale=255.0,
                channel_swap=(2,1,0))

        imagenet_labels_filename = caffe_root + 'synset_words.txt'

    inputs = [caffe.io.load_image(input_image)]

    scores = classifier.predict(inputs, True).flatten()
    indices = (-scores).argsort()[:5]
    print scores[indices[0]]
    if scores[indices[0]] < 0.3:
        return None

    # return scores[indices[1]]
    ####### Try classifier way ###########

    # load labels
    # imagenet_labels_filename = caffe_root + 'examples/alphabet/char74k/synset_words.txt'
    try:
        labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')
    except:
        print 'synset_words.txt cannot be found'

    # sort top k predictions from softmax output
    # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    # print labels[top_k]
    return labels[indices][0]

def findnext(i, words, debug=0):
    last_word = words[i]

    ref_width = last_word[4] #w
    ref_x = last_word[2] #x
    ref_midpoint_x = ref_x + ref_width/2 #midpoint
    ref_adjust = ref_width *  word_gap #0.15
    ref_midpoint_x = ref_midpoint_x + ref_adjust

    ref_y = last_word[3] #y
    ref_height = last_word[5]
    ref_midpoint_y = ref_y + last_word[5]/2 #h
    ref_centroid_y = last_word[1]

    k = 0

    for word in words:

        if k == i:
            k = k + 1
            continue

        var_width = word[4] #w
        var_x = word[2] #x
        var_midpoint_x = var_x - var_width/2 #midpoint
        var_adjust = var_width * word_gap #0.15
        var_midpoint_x = var_midpoint_x - var_adjust

        var_y = word[3]
        var_height = word[5]
        var_midpoint_y = var_y - word[5]/2
        var_centroid_y = word[1]

        # ysituation = abs(ref_centroid_y - var_centroid_y) > abs(ref_midpoint_y - var_midpoint_y)
        ysituation = abs(var_centroid_y - ref_centroid_y) > ((var_height + ref_height)/2)
        xsituation =  abs(var_midpoint_x - ref_midpoint_x) < (ref_width + var_width)/2

        if debug:
                cv2.circle(img, (int(var_midpoint_x), var_y), 5, (255, 255, 255), -1)
                cv2.circle(img, (int(ref_midpoint_x), ref_y), 5, (0, 0, 0), -1)

        if ((var_midpoint_x < ref_midpoint_x) and (ysituation == False) and (xsituation == True)):
        # if ((var_midpoint_x < ref_midpoint_x) and (ysituation == False)):
            return k

        k = k + 1

    return None

def findprev(i, words, debug = 0):
    last_word = words[i]

    ref_width = last_word[4]
    ref_x = last_word[2]
    ref_midpoint_x = ref_x - ref_width/2
    ref_adjust = ref_width *  word_gap #0.15
    ref_midpoint_x = ref_midpoint_x - ref_adjust

    ref_y = last_word[3]
    ref_height = last_word[5]
    ref_midpoint_y = ref_y + last_word[5]/2
    ref_centroid_y = last_word[1]

    k = 0

    for word in words:

        if k == i:
            k = k + 1
            continue

        var_width = word[4]
        var_x = word[2]
        var_midpoint_x = var_x + var_width/2
        var_adjust = var_width * word_gap #0.15
        var_midpoint_x = var_midpoint_x + var_adjust

        var_y = word[3]
        var_height = word[5]
        var_midpoint_y = var_y - word[5]/2
        var_centroid_y = word[1]

        # ysituation = abs(var_centroid_y - ref_centroid_y) > abs(var_midpoint_y - ref_midpoint_y)
        ysituation = abs(var_centroid_y - ref_centroid_y) > ((var_height + ref_height)/2)
        
        xsituation =  abs(var_midpoint_x - ref_midpoint_x) < (ref_width + var_width)/2

        if ((var_midpoint_x > ref_midpoint_x) and (ysituation == False) and (xsituation == True)):
            return k

        k = k + 1

    return None

###########main program###############

#load rack image
rack_img = cv2.imread(input_file)

y = 2900
h = 460
x = 2600
w = 1870
orig_img = rack_img[y:y+h, x:x+w]

# Load the image
# orig_img = cv2.imread(input_file)
orig_img_alpha = cv2.imread(input_file, -1)
orig_img_alpha = cv2.copyMakeBorder(orig_img_alpha, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
alpha_mask = np.zeros(orig_img_alpha.shape, dtype=np.uint8)

# Add a border to the image for processing sake
img = cv2.copyMakeBorder(orig_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)
img_copy = img.copy()

#gray and thresh
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(gray,150,200,cv2.THRESH_BINARY_INV) # threshold
thresh = cv2.blur(thresh, (1, 1))
edges = cv2.Canny(thresh, 200, 250)

# Calculate the width and height of the image
img_y = len(img)
img_x = len(img[0])

#set global vars
img_w = img_x
img_h = img_y

#diagnostic info about original image
# print "image area: ", img_h * img_w
if (float(img_w)/img_h) > 0.7 and (float(img_w/img_h) < 1.3):
    print "image is almost square (ratio): ", float(img_w)/img_h

if DEBUG:
    print "Image is " + str(len(img)) + "x" + str(len(img[0]))

# #Split out each channel
# blue, green, red = cv2.split(img)

# # Run canny edge detection on each channel
# blue_edges = cv2.Canny(blue, 220, 250)
# green_edges = cv2.Canny(green, 220, 250)
# red_edges = cv2.Canny(red, 220, 250)

# # Join edges back into image
# edges2 = blue_edges | green_edges | red_edges

# edges = cv2.bitwise_and(edges,edges2)

# Find the contours
edgescopy, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #original is cv2.CHAIN_APPROX_NONE

# cv2.imshow("edgescopy", edgescopy)
cv2.imshow("edges", edges)
cv2.imwrite("edges.png", edges)

hierarchy = hierarchy[0]

if DEBUG:
    processed = edges.copy()
    rejected = edges.copy()

# These are the boxes that we are determining
keepers = []
kount = 0
words = []
kk = 0

letter_map = []

# For each contour, find the bounding rectangle and decide
# if it's one we care about
for index_, contour_ in enumerate(contours):
    if DEBUG:
        print "Processing #%d" % index_

    x, y, w, h = cv2.boundingRect(contour_)

    if hierarchy[kount][1] < 0:
        kount = kount + 1
        continue

    crop_img = img_copy[y:y+h, x:x+w]
    # crop_img = thresh[y:y+h, x:x+w]
    # crop_img = cv2.cvtColor(crop_img,cv2.COLOR_GRAY2RGB)
    m,n,k = crop_img.shape #rows, columns, channels //m = height, n = width
    # copy_orig_img = img[y_:y_+height, x_:x_+width]
    

    #box mark skipping logic
    if ((float(n)/m > 2) or (m < 5) or (n < 10)): #m is height, n is width
        kount = kount + 1
        continue

    flag = 0
    # Check the contour and it's bounding box
    if keep(contour_) and include_box(index_, hierarchy, contour_):
        # It's a winner!
        keepers.append([contour_, [x, y, w, h]])

        M = cv2.moments(contour_)

        if(M['m00'] > 0):
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            # cv2.circle(img_copy, (cx, cy), 5, (255, 255, 255), 2)

        #maintain key paris
        words.insert(kk, ((cx,cy, x, y, w, h)))
        kk = kk + 1
        flag = 1

    kount = kount + 1

    cv2.imwrite("crop/1312.jpg",crop_img)
    # ch = classify_char('crop/1312.jpg')
    ch = net_process_char('crop/1312.jpg')
    if ch != None:
        # print "ch: ", ch
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (100, 100, 100), 1)
        cv2.putText(img_copy, str(ch[-1]), (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0),1)
        if flag:
            letter_map.insert(kk-1, str(ch[-1]))
            flag = 0

    # draw rectangle around contour on original image
    # cv2.rectangle(img_copy,(x,y),(x+w,y+h),(255,0,255),2)

    # # Check the contour and it's bounding box
    # if keep(contour_) and include_box(index_, hierarchy, contour_):
    #     # It's a winner!
    #     keepers.append([contour_, [x, y, w, h]])
    #     #crop images of letters
    #     # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    #     # processed = edges.copy()
    #     # crop_img = processed[y:y+h, x:x+w]
    #     # cv2.imwrite("crop/crop"+str(index_)+".jpg",crop_img)

    #     if DEBUG:
    #         cv2.rectangle(processed, (x, y), (x + w, y + h), (100, 100, 100), 1)
    #         cv2.putText(processed, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    #         #try creating images
    # else:
    #     if DEBUG:
    #         cv2.rectangle(rejected, (x, y), (x + w, y + h), (100, 100, 100), 1)
    #         cv2.putText(rejected, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))


cv2.imshow("marked_img_copy", img_copy)

#create pointers for each letter
prev = []
next = []
wordlist = []

i = 0

for let1 in words:

    prev.insert(i, findprev(i, words))
    next.insert(i, findnext(i, words))
    i =i + 1

#start extracting the words
# initialize visited list
visited = [0] * len(words)

i = 0
for m, item in enumerate(words):
    # m = 43
    if visited[m]:
        continue

    k = m

    # for let1 in words:
    visited[k] = 1
    let1 = words[k]
    right_fragment = ""
    left_fragment = ""

    cv2.circle(img_copy, (let1[0], let1[1]), 5, (0, 0, 255), -1)
    while next[k] != None:
        # print "inside next"
        k = next[k]
        let1 = words[k]
        visited[k] = 1
        cv2.circle(img_copy, (let1[0], let1[1]), 5, (0, 255, 0), -1)
        right_fragment = right_fragment + letter_map[k]

    # right_fragment = right_fragment[::-1]
    k = m

    while prev[k] != None:
        # print "inside prev"
        k = prev[k]
        let1 = words[k]
        visited[k] = 1
        cv2.circle(img_copy, (let1[0], let1[1]), 5, (0, 255, 0), -1)
        left_fragment = left_fragment + letter_map[k]

    left_fragment = left_fragment[::-1]

    found_word = left_fragment+letter_map[m]+right_fragment
    if len(found_word) > 1:
        print "found word: ", found_word
        wordlist.insert(i, found_word)
        i = i + 1

#call dictionary
# wordlist = dbcall(wordlist, 0.3)

# Make a white copy of our image
new_image = edges.copy()
new_image.fill(255)
boxes = []

# For each box, find the foreground and background intensities
for index_, (contour_, box) in enumerate(keepers):

    # Find the average intensity of the edge pixels to
    # determine the foreground intensity
    fg_int = 0.0
    for p in contour_:
        fg_int += ii(p[0][0], p[0][1])

    fg_int /= len(contour_)
    if DEBUG:
        print "FG Intensity for #%d = %d" % (index_, fg_int)

    # Find the intensity of three pixels going around the
    # outside of each corner of the bounding box to determine
    # the background intensity
    x_, y_, width, height = box
    bg_int = \
        [
            # bottom left corner 3 pixels
            #also check mid points too!
            ii(x_ - 1, y_ - 1),
            ii(x_ - 1, y_),
            ii(x_, y_ - 1),
            ii(x_ - width/2 - 1, y_ - height/2 - 1),
            ii(x_ - width/2 - 1, y_ - height/2),
            ii(x_ - width/2, y_ - height/2 - 1),

            # bottom right corner 3 pixels
            ii(x_ + width + 1, y_ - 1),
            ii(x_ + width, y_ - 1),
            ii(x_ + width + 1, y_),
            ii(x_ + width/2 + 1, y_ - 1),
            ii(x_ + width/2, y_ - 1),
            ii(x_ + width/2 + 1, y_),

            # top left corner 3 pixels
            ii(x_ - 1, y_ + height + 1),
            ii(x_ - 1, y_ + height),
            ii(x_, y_ + height + 1),
            ii(x_ - 1, y_ + height/2 + 1),
            ii(x_ - 1, y_ + height/2),
            ii(x_, y_ + height/2 + 1),

            # top right corner 3 pixels
            ii(x_ + width + 1, y_ + height + 1),
            ii(x_ + width, y_ + height + 1),
            ii(x_ + width + 1, y_ + height),
            ii(x_ + width/2 + 1, y_ + height/2 + 1),
            ii(x_ + width/2, y_ + height/2 + 1),
            ii(x_ + width/2 + 1, y_ + height/2)
        ]

    # Find the median of the background
    # pixels determined above
    bg_int = np.median(bg_int)

    if DEBUG:
        print "BG Intensity for #%d = %s" % (index_, repr(bg_int))

    # Determine if the box should be inverted
    if fg_int >= bg_int:
        fg = 255
        bg = 0
    else:
        fg = 0
        bg = 255

    # #hardcoding test
    # fg = 0
    # bg = 255

        # Loop through every pixel in the box and color the
        # pixel accordingly
    for x in range(x_, x_ + width):
        for y in range(y_, y_ + height):
            if y >= img_y or x >= img_x:
                if DEBUG:
                    print "pixel out of bounds (%d,%d)" % (y, x)
                continue
            if ii(x, y) > fg_int:
                new_image[y][x] = bg
            else:
                new_image[y][x] = fg

# blur a bit to improve ocr accuracy
new_image = cv2.blur(new_image, (2, 2))
copy_img = new_image.copy()
copy_img = cv2.cvtColor(copy_img,cv2.COLOR_GRAY2RGB)
cv2.imwrite(output_file, new_image)
if DEBUG:
    cv2.imwrite(input_file+'_edges.png', edges)
    cv2.imwrite(input_file+'_processed.png', processed)
    cv2.imwrite(input_file+'_rejected.png', rejected)

cnt = 0
for index_, (contour_, box) in enumerate(keepers):
    x_, y_, width, height = box
    crop_img = new_image[y_:y_+height, x_:x_+width]
    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_GRAY2RGB)

    copy_orig_img = img[y_:y_+height, x_:x_+width]
    # cv2.imwrite("crop/crop"+str(index_)+".jpg",crop_img)
    # print crop_img.shape
    m,n,k = crop_img.shape #rows, columns, channels //m = height, n = width
    # if(m < 5 or n < 5):

    #box mark skipping logic
    if ((float(n)/m > 2) or (m < 10) or (n < 5)): #m is height, n is width
        continue

    # print "m, n: ", m, n
    # print "img_h, img_w: ", img_h, img_w
    # print "m/img_h: ", m/img_h
    # print "n/img_w: ", n/img_w

    # if (float(m)/img_h > 0.1 or float(n)/img_w > 0.1): #10% of height or width are ignored
    #     continue

    # #we would need different strategies to handle different shape. below one is for square or almost square.
    # if (float(img_h)/m > 10000 or float(img_w)/n > 10000): #10% of height or width are ignored
    #     continue

    cv2.imwrite("crop/1312.jpg",copy_orig_img)
    # crop_img = Image.fromarray(crop_img, 'RGB')
    # ch = classify_char('crop/1312.jpg')
    # ch = net_process_char('crop/1312.jpg')
    # print ch[-1]
    # if ch != None:
    #     # try rotated bounding box rect
    #     rrect = cv2.minAreaRect(contour_)
    #     rbox = cv2.boxPoints(rrect)
    #     rbox = np.int0(rbox)
    #     cv2.drawContours(img,[box],0,(0,0,255),2)

    #     cv2.rectangle(img, (x_, y_), (x_ + width, y_ + height), (100, 100, 100), 1)
    #     cv2.putText(img, str(ch[-1]), (x_, y_ - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),1)

    # try rotated bounding box rect
    rrect = cv2.minAreaRect(contour_)
    # area = cv2.contourArea(contour_)

    (x,y),(MA,ma),angle = cv2.fitEllipse(contour_)
    # # print "angle: ", angle
    # if angle > 110 and angle < 160:
    #     continue

    # if angle < 55 and angle > 25:
    #     continue

    # print "angle: ", angle


    # print "coutour area: ", area
    # print "area ratio: ", (area / (img_h * img_w))
    # if (area / (img_h * img_w)) > 0.1:
    #     continue

    rbox = cv2.boxPoints(rrect)
    rbox = np.int0(rbox)

    #try extracting rbox 
    roi_corners = np.array([rbox], dtype=np.int32)
    channel_count = orig_img_alpha.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(alpha_mask, roi_corners, ignore_mask_color)

    # apply the mask
    masked_image = cv2.bitwise_and(orig_img_alpha, alpha_mask)
    masked_image = masked_image[y_:y_+height, x_:x_+width]
    # save the result
    cv2.imwrite('crop/1312_image_masked_.png', masked_image)
    # cv2.imwrite('crop/image_masked_'+str(cnt)+'.png', masked_image)
    # cnt = cnt + 1

    # print "rbox: ", rbox #rbox:  [[372  97] [372  62] [392  62] [392  97]]
    # # eliminate smaller flat boxes (slope method)
    # dx = abs(rbox[0][0] - rbox[2][0]) # x-distance
    # dy = abs(rbox[0][1] - rbox[1][1]) # y-distance

    # # print "dx, dy: ", dx, dy
    # if(dx == 0 or dy == 0):
    #     continue

    # if(float(dx)/dy) < 0.2 or (float(dy)/dx < 0.2):
    #     continue

    # now check the distance between points
    #[[372  97]       [372  62]        [392  62]       [392  97]]
    #[[0][0] [0][1]]  [[1][0] [1][1]]  [[2][0] [2][1]]
    linedist1 = math.sqrt(((rbox[0][0] - rbox[1][0]) * (rbox[0][0] - rbox[1][0])) + ((rbox[0][1] - rbox[1][1]) * (rbox[0][1] - rbox[1][1])))
    linedist2 = math.sqrt(((rbox[2][0] - rbox[1][0]) * (rbox[2][0] - rbox[1][0])) + ((rbox[1][1] - rbox[2][1]) * (rbox[1][1] - rbox[2][1])))

    # print "linedist1, linedist2 ", linedist1, linedist2
    if linedist1 == 0 or linedist2 == 0:
        continue

    if( float(linedist2)/linedist1 < 0.2 or float(linedist1)/linedist2 < 0.2):
        continue

    # cv2.drawContours(img,[rbox],0,(0,255,0),2)
    # # draw contours with different colors based on thier orientation
    # if angle > 100 and angle < 170:
    #     # cv2.drawContours(img,[rbox],0,(255,0,0),2)
    #     continue
    # elif angle < 30 and angle > 70:
    #     # cv2.drawContours(img,[rbox],0,(0,0,255),2)
    #     continue
    # else:
    #     cv2.drawContours(img,[rbox],0,(0,255,0),2)

    # cv2.rectangle(img, (x_, y_), (x_ + width, y_ + height), (100, 100, 100), 1)
    # cv2.putText(img, str(ch[-1]), (x_, y_ - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),1)

cv2.imwrite("Text_Marked_1312.jpg",img_copy)

# cv2.namedWindow("labelled_image", cv2.WINDOW_AUTOSIZE) 
# cv2.resizeWindow("labelled_image", 640, 480)         
# cv2.setWindowProperty("labelled image", cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
cv2.imshow('labelled_image',img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
