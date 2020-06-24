from flask import Flask, Response, render_template, request, jsonify
import logging
from emotionDetector import get_emotion
from config import *
import numpy as np
import cv2
import json
import argparse

app = Flask(__name__)

# Global variabble
SURGICAL_MASK = 1
SHOWMASK = True
CUR_MASK = SURGICAL_MASK

@app.route('/')
def index():
    return "Hi there"


@app.route('/emotion', methods=['POST'])
def emotion():
    print("Emotion")
    # npimg = np.fromstring(request.files['file'].read(), np.uint8)
    npimg = np.frombuffer(request.files['file'].read(), np.uint8)
    data = json.loads(request.form['json'])
    landmark = data['landmark']
    print("get landmark")
    print(landmark)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite("saved.jpg", img)
    print(img.shape)
    ##############################
    emotion = get_emotion(img, None)
    # To the preprocessing here
    ##############################
    response = jsonify({'emotion': emotion})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    return response

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", required=False, type=int, default=7007, help="Write your desired port number")
    args = vars(ap.parse_args())
    app.run(host='0.0.0.0', port=args['port'])