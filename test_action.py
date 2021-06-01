#import necessary package
import requests
import cv2
import sys
import argparse
import numpy as np

url = 'http://127.0.0.1:8000/api/action_recogniser/'

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--link", type =str, default ="",  help ="optional link to youtube video")
ap.add_argument("-p", "--path", type =str, default="",  help ="optional path to video file")
args = vars(ap.parse_args())

if not args['link']:
    link = None
    if not args['path']:
        link = None
else:
    link = args['link']

payload = {'url' : link}

#send the received payload to server
r = requests.post(url, data = payload).json()

# read the individual frame and display it
for fr in r['frames']:
    fr = np.asarray(fr).astype('uint8')
    cv2.imshow("action recognition", fr)
    cv2.waitKey(1)
