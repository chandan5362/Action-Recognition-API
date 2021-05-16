import numpy as np
import argparse
import imutils
import sys
import cv2
import os
from pytube import YouTube

ap = argparse.ArgumentParser()

# ap.add_argument("-l", "--link", required =True, help ="provide youtube video link")
# ap.add_argument("-c", "--classes", required =True, help ="path to labeled text file")
ap.add_argument("-l", "--link", type =str, default ="",  help ="optional path to video file")

args = vars(ap.parse_args())


#initilize video downloader

try:
  yt = YouTube(args["link"])
except:
  print("connection error")

#download youtube video
#store the video in current directory
try:
  yt = yt.streams.filter(file_extension = 'mp4')[0]
  st = yt.download()
  # st = yt.streams.first().download()
  os.rename(st, 'test_video.mp4')
except:
  print("download error")
  print("playing the default video")

#load contents of class label file and define the duration  and sample size of the video.
classes = "./labels.txt"
CLASSES = open(classes).read().strip().split("\n")
frame_per_clip = 16
sample_size = 112 

#load the model
model = "./resnet-34_kinetics.onnx"
net = cv2.dnn.readNet(model)
print("Model loaded successfully")

print("[INFO] accessing video stream")
# input_video = "
title = './test_video.mp4' if not args['link']== "" else './sample.avi'
# print(title)
stream = cv2.VideoCapture( title if title else 0)
# stream = cv2.VideoCapture( args["input"] if args["input"]  else 0)

#loop untill we finish loading
while True:
  frames = []

  for i in range(frame_per_clip):
    (grabbed, frame) =  stream.read()

    if not grabbed:
      print("No frame to read --exiting")
      sys.exit(0)

    #resize the read frame and add it to the frame list
    frame = imutils.resize(frame, width = 400)
    frames.append(frame)

  #construct blob from the frame list
  #image preprocessing
  blob = cv2.dnn.blobFromImages(frames, 1.0, (sample_size, sample_size), (114.7748, 107.7354, 99.4750), swapRB = True, crop = True)
  blob = np.transpose(blob, (1, 0, 2, 3))
  blob = np.expand_dims(blob, axis = 0)

  #pass the blob through the network
  net.setInput(blob)
  op = net.forward()
  label = CLASSES[np.argmax(op)] #print the output with maximum probability

  #draw the activity onto the frame
  for frame in frames:
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    #display the frame onto screen
    cv2.imshow("Action Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    if key ==ord("q"):
      break



