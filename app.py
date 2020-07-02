from flask import Flask, Response, render_template, request, jsonify
import logging
# from emotionDetector import get_emotion
from config import *
import numpy as np
import cv2
import json
import numpy as numpy
import timeit
import argparse
import os
import tensorflow as tf
import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



def get_datagen(dataset, num, aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            target_size=(197, 197),
            color_mode='rgb',
            shuffle = False,
            class_mode='categorical',
            batch_size=num)

model = load_model("./models/cnn.h5")
NUM_FRAME = 30
frame_list = []

def get_emotion(img, statistics = False):
    global frame_list
    frame_list.append(img)
    #emotions = ['unsatisfied', 'astonished', 'joyful', 'sadness', 'neutral']
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    if len(frame_list) < NUM_FRAME:
        return NEUTRAL, NEUTRAL

    elif len(frame_list) >= NUM_FRAME:
        
        for i in range(NUM_FRAME):
            cv2.imwrite('frame/frame/'+str(i)+ '.png', frame_list[i])

        test_generator = get_datagen('frame', NUM_FRAME)
        output= model.predict_generator(test_generator, steps=1)

        ychats = np.array(output)
        summed = np.sum(ychats, axis=0)
        outcomes = np.argmax(ychats, axis=1)
        frame_list = frame_list[1:NUM_FRAME]

        if statistics:
            results = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

            for out in outcomes:
                results[out] += 1

            results = np.divide(results, NUM_FRAME)
            results = np.multiply(results, 100)
            stat = {"unsatisfied": results[0]+results[1],
                    "astonished": results[2]+results[5],
                    "joyful": results[3],
                    "sadness": results[4],
                    "neutral": results[6]}
            return stat

        summed = np.divide(summed, NUM_FRAME)
        stat = {"unsatisfied": summed[0]+summed[1],
                    "astonished": summed[2]+summed[5],
                    "joyful": summed[3],
                    "sadness": summed[4],
                    "neutral": summed[6]}

        m = max(summed)
        index = [i for i, j in enumerate(summed) if j == m]

        frequent = emotions[index[0]]

        if frequent == 'angry' or frequent == 'disgust':
            frequent =  UNSATISFIED
        elif frequent == 'fear'  or frequent == 'surprise':
            frequent =  ASTONISHED
        elif frequent == 'sad':
            frequent =  SAD
        elif frequent == 'happy':
            frequent =  JOYFUL
        else:
            frequent = NEUTRAL

        results = np.array([0, 0, 0, 0, 0, 0, 0])
        for out in ychats:
            m = max(out)
            if m > 0.85:
                index = [i for i, j in enumerate(out) if j == m]
                results[index[0]] += 1
        
        m = max(results)
        if m != 0 :
            index = [i for i, j in enumerate(results) if j == m]
            likely = emotions[index[0]]
                
            if likely == 'angry' or likely == 'disgust':
                likely =  UNSATISFIED
            elif likely == 'fear'  or likely == 'surprise':
                likely =  ASTONISHED
            elif likely == 'sad':
                likely =  SAD
            elif likely == 'happy':
                likely =  JOYFUL
            else:
                likely = NEUTRAL
        else:
            likely = NEUTRAL

        return frequent, likely


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
    start_time = timeit.default_timer()
    print("received data")
    # npimg = np.fromstring(request.files['file'].read(), np.uint8)
    npimg = np.frombuffer(request.files['file'].read(), np.uint8)
    # data = json.loads(request.form['json'])
    # landmark = data['landmark']
    # print("get landmark")
    # print(landmark)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.imwrite("saved.jpg", img)
    print(img.shape)
    ##############################
    emotion, _ = get_emotion(img)
    # print("Result is %s" % emotion)
    # To the preprocessing here
    ##############################
    response = jsonify({'emotion': emotion})
    response.headers.add('Access-Control-Allow-Origin', '*')
    app.logger.info(response)
    elapsed_time = timeit.default_timer() - start_time
    print("result is %d and call took %fs to process" % (emotion, elapsed_time))
    return response

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--port", required=False, type=int, default=7007, help="Write your desired port number")
    args = vars(ap.parse_args())
    app.run(host='0.0.0.0', port=args['port'], debug=False, threaded=False)
