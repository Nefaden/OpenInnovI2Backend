import os
import json
import IPython.display as ipd
# % pylab inline
import pandas as pd
import librosa
import glob 
import librosa.display
import random
import math
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.externals.joblib import dump, load
from joblib import dump, load
from sklearn import metrics 

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

import tensorflow_io as tfio
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from datetime import datetime
from numba import jit, cuda
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import pulsar
from pulsar.schema import *

# Shall be moved to server config later
from dotenv import load_dotenv
load_dotenv()

# loading the model into the session created above
model = load_model(os.getenv("MODEL"))
stmTime = pathlib.Path(os.getenv("MODEL")).stat().st_mtime
service_url = os.getenv("PULSAR_URL")
topic = os.getenv("PULSAR_PREDICT_TOPIC")
subscription = os.getenv("PULSAR_SUBSCRIPTION")
ack_grouping_time = os.getenv("PULSAR_GROUP_TIME")
token=os.getenv("PULSAR_TOKEN")
# receiver_q_size = os.getenv("PULSAR_Q_SIZE")
receiver_q_size = 10

class File(Record):
    path = String()
    name = String()
    user = String()

class Prediction(Record):
    path = String()
    name = String()
    speaker = Integer()

#Pulsar = tfio.experimental.streaming.PulsarIODataset(service_url,topic,subscription,100,1,100)
#tfio.IODataset()
client = pulsar.Client(service_url, pulsar.AuthenticationToken(token))
consumer = client.create_reader(topic,pulsar.MessageId.latest, receiver_queue_size=receiver_q_size, reader_name=subscription, schema=AvroSchema(File))
producer = client.create_producer(topic)
# directory = os.fsencode('voices/')

# Change speaker with the created room's ID (queue)
# for file in os.listdir(directory):
#       flac_name = os.fsdecode(file)
#       producer.send(File(path='voices/' + flac_name, name=flac_name, speaker=103))

# Although this function was modified and many parameteres were explored with, most of it
# came from Source 8 (sources in the READ.ME)

def extract_features(files):    
    # Sets the name to be the path to where the file is in my computer
    file_name = os.path.join(files.path)
    
    # Loads the audio file as a floating point time series and assigns the default sample rate
    # Sample rate is set to 22050 by default
    X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    
    # Generate Mel-frequency cepstral coefficients (MFCCs) from a time series 
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)

    # Generates a Short-time Fourier transform (STFT) to use in the chroma_stft
    stft = np.abs(librosa.stft(X))

    # Computes a chromagram from a waveform or power spectrogram.
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)

    # Computes a mel-scaled spectrogram.
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)

    # Computes spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)

    # Computes the tonal centroid features (tonnetz)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

model.summary()

labels = np.load(os.getenv("CLASSES"))
ss = load(os.getenv("SCALER"))

lb = LabelEncoder()
lb.fit(labels)
while True:
  msg = consumer.read_next()
  st = pathlib.Path(os.getenv("MODEL")).stat().st_mtime
  if st > stmTime :
    stmTime = st
    model = load_model(os.getenv("MODEL")) 
    labels = np.load(os.getenv("CLASSES"))
    ss = load(os.getenv("SCALER"))
  model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam') 
  content = msg.value()
  file = pd.DataFrame([[content.path, content.name]], columns= ['path','file'])
  features_label = file.apply(extract_features, axis=1)
  # We create an empty list where we will concatenate all the features into one long feature
  # for each file to feed into our neural network 
  print(features_label[0])
  features = []
  for i in range(0, len(features_label)):
      features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                  features_label[i][2], features_label[i][3],
                  features_label[i][4]), axis=0))
  X = np.array(features)
  
  X = ss.transform(X)
  print(len(features))
  preds = np.argmax(model.predict(X), axis=-1)
  print(preds)
  preds = lb.inverse_transform(preds)
  print(preds)
  producer = client.create_producer(content.user, schema=JsonSchema(Prediction))
  producer.send(Prediction(path=content.path, name=content.name, speaker=int(preds[0])))
