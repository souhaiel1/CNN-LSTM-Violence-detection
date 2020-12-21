from __future__ import absolute_import
from __future__  import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import numpy as np
import os
from violencemodel import *
from flask import Flask , request , jsonify , Response
from PIL import Image
from io import BytesIO
import time
from skimage.transform import resize
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import  Dropout, Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
from twilio.rest import Client

model1 = souhaiel_model(tf)
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")
args = vars(ap.parse_args())

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = model1
# initialize the image mean for mean subtraction along with the
# predictions queue
#mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
vc=cv2.VideoCapture('weeknd1.mp4')
fps = vs.get(cv2.CAP_PROP_FPS)
writer = None
(W, H) = (None, None)
#client = Client("ACea4cecca40ebb1bf4594098d5cef4541", "32789639585561088d5937514694e115") #update from twilio
prelabel = ''
ok = 'Normal'
okk='violence'
i=0
frames = np.zeros((30, 160, 160, 3), dtype=np.float)
datav = np.zeros((1, 30, 160, 160, 3), dtype=np.float)
frame_counter=0

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frm) = vc.read()
    
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frm.shape[:2]
    #framecount = framecount+1
    # clone the output frame, then convert it from BGR to RGB
    # ordering, resize the frame to a fixed 224x224, and then
    # perform mean subtraction
    output = frm.copy()
    while i < 30:
        rval, frame = vs.read()
        frame_counter +=1
        if frame_counter == vs.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = resize(frame,(160,160,3))
        frame = np.expand_dims(frame,axis=0)
        if(np.max(frame)>1):
            frame = frame/255.0
        frames[i][:] = frame
        i +=1
        
    datav[0][:][:] =frames
    frames -= mean
    

	# make predictions on the frame and then update the predictions
	# queue
    #preds = model1.predict(datav)
#	print('Preds = :', preds)
	
#	total = (preds[0]+ preds[1]+preds[2] + preds[3]+ preds[4]+preds[5])
#	maximum = max(preds)
#	rest = total - maximum
    
#	diff = (.8*maximum) - (.1*rest)
#	print('Difference of prob ', diff)
#	th = 100
#	if diff > .60:
#		th = diff
#	print('Old threshold = ', th)
    
    
    prediction = preds.argmax(axis=0)
    Q.append(preds)

	# perform prediction averaging over the current history of
	# previous predictions
    results = np.array(Q).mean(axis=0)
    print('Results = ', results)
    maxprob = np.max(results)
    print('Maximun Probability = ', maxprob)
    i = np.argmax(results)
    rest = 1 - maxprob
    
    diff = (maxprob) - (rest)
    print('Difference of prob ', diff)
    th = 100
    if diff > .80:
        th = diff
      
        
        
        
    if (preds[0][1]) < th:
        text = "Alert : {} - {:.2f}%".format((ok), 100 - (maxprob * 100))
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
    else:
		
        text = "Alert : {} - {:.2f}%".format((okk), maxprob * 100)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 5) 
#		if label != prelabel:
#			client.messages.create(to="<+country code>< receiver mobile number>", #for example +918255555555
#                       from_="+180840084XX", #sender number can be coped from twilo
#                       body='\n'+ str(text) +'\n Satellite: ' + str(camid) + '\n Orbit: ' + location)
    


	# check if the video writer is None
    if writer is None:
	    # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 27.0,
			(W, H), True)

	# write the output frame to disk
    writer.write(output)

	# show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
#print('Frame count', framecount)
# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
vc.release()
