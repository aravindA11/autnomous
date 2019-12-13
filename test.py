import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import utils

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from concurrent import futures
import time
import logging
import socket

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

import cv2
#cap = cv2.VideoCapture(1)
sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
# What model to download.
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'#'tennis_graph'#ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/hammer/Downloads/models/research/object_detection/Trained RCNN Model for TennisBall/training/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('Trained RCNN Model for TennisBall', 'object-detection.pbtxt')

NUM_CLASSES = 1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

session = tf.Session(config=config)




detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)

def objDet():
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            #while True:
                cap = cv2.VideoCapture(2)
                _, image_np = cap.read()
                #                               Do not change, to be tested
                splitSize = 2
                height, width = image_np.shape[:2]
                splitter = float(splitSize)
                termH = float(height/splitter)
               # print (termH,"   ",height,"   ",splitter)
                termW = float(width/splitter)
               # print(height,"--,--",width)
                for i in range(0,splitSize-1):       
                    for j in range (0,splitSize-1):
                       # print ( int(j*termH ),",",int(i*termW),"    ",int((j+1)*termH),",",(i+1)*termW  )
                       # image_np = image_np[ int( (j)*termH) : int(((j+1))*termH )-1 , int( (i)*termW): int(((i+1))*termW ) -1]
                        xoffset = i*termW

                        image_np_expanded = np.expand_dims(image_np, axis=0)
                        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                        # Each box represents a part of the image where a particular object was detected.
                        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                        # Each score represent how level of confidence for each of the objects.
                        # Score is shown on the result image, together with the class label.
                        scores = detection_graph.get_tensor_by_name('detection_scores:0')
                        classes = detection_graph.get_tensor_by_name('detection_classes:0')
                        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                        # Actual detection.
                        (boxes, scores, classes, num_detections) = sess.run(
                            [boxes, scores, classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)

                        im_height, im_width = image_np.shape[:2]
                        position = boxes[0][0]
                        (xmin, xmax, ymin, ymax) = (position[1]*im_width, position[3]*im_width, position[0]*im_height, position[2]*im_height)
                        roi = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]
                       # print ((xmin + xmax)/2)
                       # print (xmax-xmin)
                        #mid=(xmin + xmax)/2
                        cv2.imshow('frame',image_np)
                        cv2.waitKey(10)
                        dataString=str(xoffset+xmin)+","+str(xoffset+xmax)
                        #if(xmax!=0):
                        print(xmax-xmin)
                        return dataString
                # for i in range ( 0 , splitSize -1):
                #     for j in range (0,splitSize -1):
                #         image_np = image_np[ int( (j)*termH + (termH/2) ): int((j+1)*termH +(termH/2)) , int((i)*termW + (termW/2)) :int( ((i+1))*(termW) +(termW/2 )) ]
 
                # #
                #         #cv2.imshow('abc',image_np)
                #         # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                #         image_np_expanded = np.expand_dims(image_np, axis=0)
                #         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                #         # Each box represents a part of the image where a particular object was detected.
                #         boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                #         # Each score represent how level of confidence for each of the objects.
                #         # Score is shown on the result image, together with the class label.
                #         scores = detection_graph.get_tensor_by_name('detection_scores:0')
                #         classes = detection_graph.get_tensor_by_name('detection_classes:0')
                #         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                #         # Actual detection.
                #         (boxes, scores, classes, num_detections) = sess.run(
                #             [boxes, scores, classes, num_detections],
                #             feed_dict={image_tensor: image_np_expanded})
                #         # Visualization of the results of a detection.
                #         vis_util.visualize_boxes_and_labels_on_image_array(
                #                 image_np,
                #                 np.squeeze(boxes),
                #                 np.squeeze(classes).astype(np.int32),
                #                 np.squeeze(scores),
                #                 category_index,
                #                 use_normalized_coordinates=True,
                #                 line_thickness=8)

                #         im_height, im_width = image_np.shape[:2]
                #         position = boxes[0][0]
                #         (xmin, xmax, ymin, ymax) = (position[1]*im_width, position[3]*im_width, position[0]*im_height, position[2]*im_height)
                #         roi = image_np[int(ymin):int(ymax), int(xmin):int(xmax)]
                #         print ((xmin + xmax)/2)
                #         print (xmax-xmin)
                #         #mid=(xmin + xmax)/2
                #         dataString=str(xoffset+xmin)+","+str(xoffset+xmax)
                #         if(xmax!=0):
                #             return dataString
                #detection_graph = tf.Graph()

                    
    return "0,0"

if __name__ == '__main__':
    """    #set the address, port,...
    port = 23910
    address = "127.0.0.1"
    s = socket.socket()
    s.bind((address, port))
    s.listen(500000)
    c, addr = s.accept()
    print("Going to the loop")
    while True:
        data, address = sock.recvfrom(4096)
    	c, addr = s.accept()
    	print ('---------Got connection from------', addr)
	print (data)
        if data=="Ready":
            #n=objDet()
            n="0,0"
            sent = sock.sendto(n, address)
            c.send(n.encode())
   """
    while 1:
      a=objDet()
      print ("Here")
      print (a)
#c.close()

