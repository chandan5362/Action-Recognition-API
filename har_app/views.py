# import the necessary packages
from django.http.response import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

import numpy as np   
import urllib
import json
import cv2
import os
import imutils
from pytube import YouTube

ACTION_RECOGNISER = '{base_path}/models/resnet-34_kinetics.onnx'.format(base_path = os.path.abspath(os.path.dirname(__file__)))
classes = '{base_path}/models/labels.txt'.format(base_path = os.path.abspath(os.path.dirname(__file__)))
SAMPLE_VIDEO = '{base_path}/models/sample.mp4'.format(base_path = os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def recogniser(request):
    data = {'success':False}

    # check to see if this is a post request
    if request.method == "POST":

        url = request.POST.get("url", None)
        path = request.POST.get("path", None)

        if path is not None:
            stream  = _grab_frames(path = path)

        if url is not None:
            stream = _grab_frames(url = url)
        # if the URL is None, then return an error
        else :
            data["error"] = "No URL provided."
            data["alternate"] = "working on default video"
            # print(SAMPLE_VIDEO)
            stream = _grab_frames(video = SAMPLE_VIDEO)
            
            
        if stream is None:
            data['error'] = "No frame captured"
            return JsonResponse(data)

        #perform the basic action recogntion opencv stuff
        #load the resner_34 model

        CLASSES = open(classes).read().strip().split("\n")
        frame_per_clip = 16
        sample_size = 112
        frames_bbox = []
        while True:
            frames = []

            for i in range(frame_per_clip):
                (grabbed, frame) =  stream.read()
                print(len(frames_bbox))
                if not grabbed:
                    data.update({'frames':frames_bbox, 'success' : True})
                    return JsonResponse(data)

                #resize the read frame and add it to the frame list
                frame = imutils.resize(frame, width = 400)
                frames.append(frame)

            #construct blob from the frame list
            #image preprocessing
            blob = cv2.dnn.blobFromImages(frames, 1.0, (sample_size, sample_size), (114.7748, 107.7354, 99.4750), swapRB = True, crop = True)
            blob = np.transpose(blob, (1, 0, 2, 3))
            blob = np.expand_dims(blob, axis = 0)

            #pass the blob through the network
            net = cv2.dnn.readNet(ACTION_RECOGNISER)
            net.setInput(blob)
            op = net.forward()
            label = CLASSES[np.argmax(op)] #print the output with maximum probability

            #draw the activity onto the frame
            for frame in frames:
                cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
                cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                #append the frame into data
                frames_bbox.append(frame.tolist())

        # data.update({'frames':frames_bbox, 'success' : True})
    # return a JSON responseS
    return JsonResponse(data)
           


def _grab_frames( path = None, url=None, video  = None):
    if path is not None:
        stream = cv2.VideoCapture(path)
    elif url is None:
        stream = cv2.VideoCapture(video)
        # print(stream)
    #otherwise video does not reside on the disk
    else:
        if url is not None:
            try:
                print(type(url))
                yt = YouTube(url)
                print(yt)
            except:
                stream = None
                print("Video not read")
            
            #download youtube video
            #store the video in current directory
            try:
                yt = yt.streams.filter(file_extension = 'mp4')[0]
                print(yt)
                st = yt.download(output_path = '{base_path}/models/'.format(base_path = os.path.abspath(os.path.dirname(__file__))))
                os.rename(st, '{base_path}/models/test_video.mp4'.format(base_path = os.path.abspath(os.path.dirname(__file__))))
            except:
                stream = None
            source = '{base_path}/models/test_video.mp4'.format(base_path = os.path.abspath(os.path.dirname(__file__)))
            ffmpeg_extract_subclip(source, 10, 30, targetname='{base_path}/models/test.mp4'.format(base_path = os.path.abspath(os.path.dirname(__file__))))
            title = '{base_path}/models/test.mp4'.format(base_path = os.path.abspath(os.path.dirname(__file__)))
            stream = cv2.VideoCapture(title)
            
    return stream


        
                 
