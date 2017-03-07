import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2
import pickle

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.convolutional import UpSampling2D

from keras.models import load_model
from keras.models import Model
from keras.models import Sequential

from sklearn.externals import joblib

import time
import utils

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
encoder = None
scaler = None

MAX_SPEED = 15
MIN_SPEED = 8
TARGET_SPEED = 5
sequence_length = 3
sequence_interval = 3

speed_limit = MAX_SPEED


def convertToTemporalData(train):
    result = []
    for index in range(len(train) - sequence_length * sequence_interval):
        sequence = []
        for sequence_index in range(sequence_length):
            sequence.append(train[index + sequence_index * sequence_interval, :])
        result.append(sequence)
    result = np.array(result)
    return result

def X_scaling(scaler, xvalues, ysize):
    """
    @param scaler: a scaler object, implementing the transform() and inverse_transform() methods
                   as for the scikit-learn StandardScaler.
                   The scaler must expects the inputs in the [X, Y] order (Y placed at the end)
    @param xvalues: the X values to scale with transform() method
    @type xvalues: array of shape (m, n)
    @param ysize: the number of Y outputs (columns)
    @type ysize: int
    @return the X values scaled
    @rtype: array of shape (m, n)
    """
    # add dummy y values
    values = np.concatenate((xvalues, np.zeros((xvalues.shape[0], ysize))), axis=1)
    scaled_values = scaler.transform(np.array(values))
    if len(xvalues) == 1:
        scaled_values = scaled_values.reshape(1, -1)
    # remove y scaled values
    return scaled_values[:, :-ysize]

    
def X_inverse_scaling(scaler, xvalues, ysize):
    """
    @param scaler: a scaler object, implementing the transform() and inverse_transform() methods
                   as for the scikit-learn StandardScaler.
                   The scaler must expects the inputs in the [X, Y] order (Y placed at the end)
    @param xvalues: the X values to scale with transform() method
    @type xvalues: array of shape (m, n)
    @param ysize: the number of Y outputs (columns)
    @type ysize: int
    @return the X values scaled
    @rtype: array of shape (m, n)
    """
    # add dummy y values
    values = np.concatenate((xvalues, np.zeros((xvalues.shape[0], ysize))), axis=1)
    scaled_values = scaler.inverse_transform(values)
    # remove y scaled values
    return scaled_values[:, :-ysize]


def Y_inverse_scaling(scaler, yvalues, xsize):
    """
    @param scaler: a scaler object, implementing the transform() and inverse_transform() methods
                   as for the scikit-learn StandardScaler.
                   The scaler must expects the inputs in the [X, Y] order (Y placed at the end)
    @param yvalues: the Y values to inverse scale with inverse_transform() method
    @type yvalues: array of shape (m, n)
    @param xsize: the number of X features (columns)
    @type xsize: int
    @return the Y values inverse scaled
    @rtype: array of shape (m, n)
    """
    # add dummy X values
    values = np.concatenate((np.zeros((yvalues.shape[0], xsize)), yvalues), axis=1)
    inv_scaled_values = scaler.inverse_transform(values)
    return inv_scaled_values[:, xsize:]

    
def createEncoder():
    input_img = Input(shape=(1, utils.ROWS, utils.COLS))
    
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(input_img)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    encoded = MaxPooling2D(pool_size=(2, 2), border_mode='same', dim_ordering='th')(x)
    
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(encoded)
    x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', dim_ordering='th')(x)
    x = UpSampling2D(size=(2, 2), dim_ordering='th')(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', dim_ordering='th')(x)
    
    encoder = Model(input=input_img, output=encoded)
    autoencoder = Model(input=input_img, output=decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    encoder.summary()
    return encoder

def createModel():
    model = Sequential()
    """
    model.add(LSTM(input_shape=(sequence_length, 128 * 15 * 40),
                        output_dim=100, activation='tanh',
                        dropout_W=dropout_W, dropout_U=dropout_U,
                        return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(output_dim=100, activation='tanh',
                  dropout_W=dropout_W, dropout_U=dropout_U))
    model.add(Dropout(dropout))
    """
    model.add(Flatten(input_shape=encoder.layers[-1].output_shape[1:],))
    model.add(Dense(output_dim=100,
                    activation='tanh',))
    model.add(Dense(output_dim=100, activation='tanh'))
    model.add(Dense(output_dim=1, activation='linear'))
    model.compile(loss='mse', optimizer="sgd")
    model.summary()
    return model
    
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        #steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            start = time.time()
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing

            image = image.reshape(-1, utils.ROWS * utils.COLS)
            # scale X
            scaled_values = X_scaling(scaler, image, 1)
            # reshape to image for CNN input
            scaled_values = scaled_values.reshape(-1, 1, utils.ROWS, utils.COLS)
            
            image = encoder.predict(scaled_values, batch_size=1)
        
            steering_angle = model.predict(image, batch_size=1)
            steering_angle = Y_inverse_scaling(scaler, steering_angle, utils.ROWS * utils.COLS)
            steering_angle = steering_angle[0][0]
            
            #steering_angle = steering_angle / 20.
            steering_angle = max(steering_angle, -1.)
            steering_angle = min(steering_angle, 1.)
            
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            """
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2
            """
            if speed >= TARGET_SPEED:
                throttle = 0.
            else:
                throttle += 0.01
                
            throttle = max(throttle, -1.)
            throttle = min(throttle, 1.)

            print('{} {} {}'.format(steering_angle, throttle, speed))
            end = time.time()
            print("prediction duration = {} ms".format((end - start) * 1000.))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    encoder = createEncoder()
    encoder.load_weights("encoder_weights_trainingSet_2017021x.h5")

    scaler = joblib.load("scaler.pkl")
    #print("scaler.mean_={}".format(scaler.mean_))
    #print("scaler.scale_={}".format(scaler.scale_))
    
    model = createModel()
    model.load_weights(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
