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
from joblib import dump, load
from sklearn import metrics

import numpy as np

from keras.utils.np_utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.backend import manual_variable_initialization

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

from datetime import datetime
from numba import jit, cuda
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import re

import pulsar
from pulsar.schema import *

from dotenv import load_dotenv
load_dotenv()

#list the files
folderlist = os.listdir('voices/') 
#read them into pandas
df = pd.DataFrame([])
for folder in folderlist:
  files = glob.glob('voices'+folder+'/**/*.flac',recursive=True)
  dtf = pd.DataFrame(files);
  dtf = dtf.rename(columns={0:'path'})
  dtf['file'] = [os.path.basename(val) for val in dtf['path']]
  dtf['speaker'] = folder
  df = pd.concat([df,dtf],ignore_index=True)

df

length = len(df.index)
val_70 = math.floor((length * 70) /100)
val_90 = math.floor((length * 90) /100)

print(val_70)
print(val_90)

df = df.sample(frac=1,random_state=18).reset_index(drop=True)
df_train = df[:val_70]
df_validation = df[val_70:val_90]
df_test = df[val_90:length]
df2 = df
len(df)
len(df2)

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

#value=0
#x =9
#numberoffiles = 10
#length =  math.ceil(len(df)/numberoffiles)
#while value < len(df):
#  value += len(df)/10
#  x += 1
#  maxLength = x*length
#  if x == numberoffiles :
#    value = len(df)
#    maxLength = len(df)
#  startTime = datetime.now()
#  features_label = df2[(x-1)*length:maxLength].apply(extract_features, axis=1)
#  print(datetime.now() - startTime)

  # Checking how the output looks
#  features_label
  # The next code in markdown saves the numpy array (in case our kernel restarts or 
  # anything happens, because it takes long to extract the features)

 # np.save('/content/drive/My Drive/DataSet/features_label'+str(x), features_label)

features_label = []
for fle in glob.glob("/features_label/features_label*.npy"):
    features_label = np.concatenate((features_label,np.load(fle, allow_pickle=True)),axis=0)
    print(len(features_label))

features_label

# We create an empty list where we will concatenate all the features into one long feature
# for each file to feed into our neural network 

features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))

len(features)

# We create an empty list where we will append all the speakers ids for each row of our
# dataframe by slicing the file name since we know the id is the first numbers before the hash
speaker = []
for i in range(0, len(df)):
    speaker.append(df['file'][i].split('-')[0])

# Now we create the speaker column in our dataframe and set it equal to our speaker list
df['speaker'] = speaker
# Checking the number of speakers or the number of different people in our voice data
df['speaker'].nunique()
# Setting our labels to be equal to our speaker list
labels = speaker

# Checking the size of labels and making sure it matches the size of features
len(labels)

df['speaker'].nunique()

# They look somewhat balanced with a min of 56 and a max of 166, mean of 114 
# with standard deviation of 15.89 (calculated from scipy)
np.unique(labels, return_counts=True)
X = np.array(features)
y = np.array(labels)
# Hot encoding y
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))
X.shape
y.shape
np.save(os.getenv("CLASSES"), lb.classes_)

# Choosing the first 9188 (70%) files to be our train data
# Choosing the next  2625 (20%) files to be our validation data
# Choosing the next  1312 (10%) files to be our test never before seen data
# This is analogous to a train test split but we add a validation split and we are making
# we do not shuffle anything since we are dealing with several time series, we already 
# checked before that we have balanced classes (analogous to stratify)

X_train = X[:val_70]
y_train = y[:val_70]

X_val = X[val_70:val_90]
y_val = y[val_70:val_90]

X_test = X[val_90:]
y_test = y[val_90:]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)
X_test = ss.transform(X_test)

dump(ss, os.getenv("SCALER"), compress=True)

# Build a simple dense model with early stopping with softmax for categorical classification
# We have 115 classes 
manual_variable_initialization(True)
model = Sequential()

model.add(Dense(193, input_shape=(193,), activation = 'relu'))
model.add(Dropout(0.1))

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))  

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))    

model.add(Dense(251, activation = 'softmax',name='Classes_layer'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto')

initBias = model.get_layer('Classes_layer').bias_initializer
initBias

# fitting the model with the train data and validation with the validation data
# we used early stop with patience 100 because we did not want to use early stop
# I leave the early stop regularization code in case anyone wants to use it
history = model.fit(X_train, y_train, batch_size=256, epochs=100, 
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])

# Check out our train accuracy and validation accuracy over epochs.
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Set figure size.
plt.figure(figsize=(12, 8))

# Generate line plot of training, testing loss over epochs.
plt.plot(train_accuracy, label='Training Accuracy', color='#185fad')
plt.plot(val_accuracy, label='Validation Accuracy', color='orange')

# Set title
plt.title('Training and Validation Accuracy by Epoch', fontsize = 25)
plt.xlabel('Epoch', fontsize = 18)
plt.ylabel('Categorical Crossentropy', fontsize = 18)
plt.xticks(range(0,100,5), range(0,100,5))

plt.legend(fontsize = 18);

# We get our predictions from the test data
preds = np.argmax(model.predict(X_test), axis=-1)

# We transform back our predictions to the speakers ids
preds = lb.inverse_transform(preds)

# We slice our dataframe to our test dataframe
df_test = df[val_90:]

# We create a new column called preds and set it equal to our predictions
df_test['preds'] = preds

# Checking how our test dataframe looks like now with our predictions
df_test

# Checking how many speakers we got wrong
df_test[df_test['speaker'] != df_test['preds']]

256# Checking our model accuracy
1-round(len(df_test[df_test['speaker'] != df_test['preds']])/len(df_test),3)

model.save(os.getenv("MODEL"))

# loading the model into the session created above
model = load_model(os.getenv("MODEL"))

# fitting the model with the train data and validation with the validation data
# we used early stop with patience 100 because we did not want to use early stop
# I leave the early stop regularization code in case anyone wants to use it

history = model.fit(X_val, y_val, batch_size=256, epochs=100, 
                    callbacks=[early_stop])

# We get our predictions from the test data
preds = np.argmax(model.predict(X_test), axis=-1)
# We transform back our predictions to the speakers ids
preds = lb.inverse_transform(preds)
# We slice our dataframe to our test dataframe
df_test = df[val_90:]
# We create a new column called preds and set it equal to our predictions
df_test['preds'] = preds
# Checking how our test dataframe looks like now with our predictions
df_test

# Checking how many speakers we got wrong
df_test[df_test['speaker'] != df_test['preds']]
# Checking our model accuracy
1-round(len(df_test[df_test['speaker'] != df_test['preds']])/len(df_test),3)

"""Idee : 
https://www.tensorflow.org/io/tutorials/kafka
"""

weights = model.get_layer('Classes_layer').get_weights()
weights

bias = model.get_layer('Classes_layer').bias.numpy()
bias

model.pop()
model.add(Dense(252, activation = 'softmax',name='Classes_layer',bias_initializer=initBias))
layer_names = [layer.name for layer in model.layers]
layer_idx = layer_names.index('Classes_layer')
weights[0].resize((128,252),refcheck=False)
weights[1].resize((252),refcheck=False)
model.layers[layer_idx].set_weights(weights)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

maxI = int(np.amax(pd.to_numeric(df['speaker']),axis=0))+1
maxI

#read them into pandas
df_perso = pd.DataFrame([])
files = glob.glob('VoicePerso/Test 1 - EC/Train/*.m4a',recursive=True)
dtf = pd.DataFrame(files);
dtf = dtf.rename(columns={0:'path'})
dtf['file'] = [os.path.basename(val) for val in dtf['path']]
dtf['speaker'] =maxI
df_perso = pd.concat([df_perso,dtf],ignore_index=True)
print(len(df_perso))
df_perso

startTime = datetime.now()
features_label_2 =df_perso.apply(extract_features, axis=1)
print(datetime.now() - startTime)
total_df = pd.concat([df,df_perso],ignore_index=True)
df_perso = pd.concat([df_test,df_perso],ignore_index=True)
features_label = np.concatenate((features_label,features_label_2),axis=0)
#np.save('/content/drive/My Drive/DataSet/features_label11', features_label)
# Checking how the output looks
#  features_label
# The next code in markdown saves the numpy array (in case our kernel restarts or 
# anything happens, because it takes long to extract the features)features_label = []
#for fle in glob.glob("/content/drive/My Drive/DataSet/features_label11.npy"):
#    features_label = np.concatenate((features_label_2,np.load(fle, allow_pickle=True)),axis=0)
#    print(len(features_label))
total_features_label = features_label
print(len(features_label))
print(len(features_label_2))
print(len(df_test))
print(len(df_perso))

# We create an empty list where we will concatenate all the features into one long feature
# for each file to feed into our neural network 

features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))

# We create an empty list where we will append all the speakers ids for each row of our
# dataframe by slicing the file name since we know the id is the first numbers before the hash
speaker = []
for i in range(0, len(df_perso)):
    speaker.append(df_perso['speaker'][i])

len(df_perso)

len(features[val_90:])

labels = speaker
labels

len(df_test)

# They look somewhat balanced with a min of 56 and a max of 166, mean of 114 
# with standard deviation of 15.89 (calculated from scipy)
np.unique(labels, return_counts=True)
X = np.array(features[val_90:])
y = np.array(labels)
# Hot encoding y
lb = LabelEncoder()
y = to_categorical(lb.fit_transform(y))
X.shape
y.shape
np.save(os.getenv("CLASSES"),lb.classes_)

X_train = ss.transform(X)

dump(ss, os.getenv("SCALER"), compress=True)

y

X

# fitting the model with the train data and validation with the validation data
# we used early stop with patience 100 because we did not want to use early stop
# I leave the early stop regularization code in case anyone wants to use it

history = model.fit(X_train, y, batch_size=256, epochs=100, 
                    callbacks=[early_stop])
model

model.save(os.getenv("MODEL"))

# We get our predictions from the test data
preds = np.argmax(model.predict(X_test), axis=-1)
# We transform back our predictions to the speakers ids
preds = lb.inverse_transform(preds)
# We slice our dataframe to our test dataframe
#df_test = df[val_90:]
# We create a new column called preds and set it equal to our predictions
df_test['preds'] = preds
# Checking how our test dataframe looks like now with our predictions
df_test

1-round(len(df_test[df_test['speaker'] != df_test['preds']])/len(df_test),3)

#read them into pandas
df_perso_test = pd.DataFrame([])
files = glob.glob('VoicePerso/Test 1 - EC/Test/*.m4a',recursive=True)
dtf = pd.DataFrame(files);
dtf = dtf.rename(columns={0:'path'})
dtf['file'] = [os.path.basename(val) for val in dtf['path']]
dtf['speaker'] =maxI
df_perso_test = pd.concat([df_perso_test,dtf],ignore_index=True)

startTime = datetime.now()
features_label =df_perso_test.apply(extract_features, axis=1)
print(datetime.now() - startTime)
#df_test = pd.concat([df_test,df_perso_test],ignore_index=True)
#features_label = np.concatenate((features_label_2,np.load(fle, allow_pickle=True)),axis=0)
features = []
for i in range(0, len(features_label)):
    features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                features_label[i][2], features_label[i][3],
                features_label[i][4]), axis=0))

speaker = []
for i in range(0, len(df_perso_test)):
    speaker.append(maxI)

labels = speaker
X = np.array(features)
X = ss.transform(X)

# We get our predictions from the test data
preds = np.argmax(model.predict(X,batch_size=32), axis=-1)
# We transform back our predictions to the speakers ids
preds = lb.inverse_transform(preds)
# We slice our dataframe to our test dataframe
# We create a new column called preds and set it equal to our predictions
df_perso_test['preds'] = preds
df_perso_test['preds'].astype(int)
# Checking how our test dataframe looks like now with our predictions
df_perso_test

df_perso_test["preds"]=df_perso_test["preds"].astype("int64")

1-round(len(df_perso_test[df_perso_test['speaker'] != df_perso_test['preds']])/len(df_perso_test),3)

df_perso_test.dtypes

df["speaker"].value_counts()

df_perso["speaker"].value_counts()

"""# Generation du code SQL pour enregistrer les données de tests en base:"""

data = df_perso["speaker"].unique()
sql_texts = []
i = 0
for code in data:
    i += 1
    if i != 252:
      sql_texts.append('INSERT INTO "actor" (first_name,last_name,url_imdb,code_ia) VALUES (\'speaker_'+str(i)+ '\','+'\'speaker_'+str(i)+'_'+str(code)+'\',null,' + str(code) +');')
    else :
      sql_texts.append('INSERT INTO "actor" (first_name,last_name,url_imdb,code_ia) VALUES (\'Eddy\', \'Cheval\',null,' + str(code) +');')
for line in sql_texts:
  print(line)

"""# Intégration de l'apprentissage dans une boucle pour Apache Pulsar :

Idée :
Le réapprentissage peut se faire partiellement comme ci dessus mais difficile de dire si le modelè n'est pas fausée.
Une méthode fournis en ligne serais de freeze les layers intermédiare pour ensuite entrainé uniquement le tous dernier layer avec l'ensemble des données
"""

class File(Record):
    path = String()
    name = String()
    speaker = Integer()

model = load_model(os.getenv("MODEL"))
stmTime = pathlib.Path(os.getenv("MODEL")).stat().st_mtime
service_url = os.getenv("PULSAR_URL")
topic = os.getenv("PULSAR_TRAINING_TOPIC")
subscription = os.getenv("PULSAR_SUBSCRIPTION")
ack_grouping_time = os.getenv("PULSAR_GROUP_TIME")
token=os.getenv("PULSAR_TOKEN")
# receiver_q_size = os.getenv("PULSAR_Q_SIZE")
receiver_q_size = 10

client = pulsar.Client(service_url,pulsar.AuthenticationToken(token))
consumer = client.create_reader(topic,pulsar.MessageId.latest,receiver_queue_size=receiver_q_size,reader_name=subscription,schema=AvroSchema(File))
producer = client.create_producer(topic,schema=AvroSchema(File))

consumer2 = client.subscribe(topic,"test",consumer_type=pulsar.ConsumerType.Exclusive,receiver_queue_size=receiver_q_size,schema=AvroSchema(File))
msg = consumer2.receive()
msg

msg = consumer2.receive()
msg.data()

for folder in folderlist:
      filelist = glob.glob('/content/drive/My Drive/DataSet/train-clean-100/'+folder+'/**/*.flac',recursive=True)
      files = []
      files.appen(filelist)
      for file in files:
        flac_name = os.fsdecode(file)
        producer.send(File(path='voices/' +folder+'/**/ ' + flac_name, name=flac_name, speaker=103))
producer.send(File(path='VoicePerso/Test 1 - EC/Test/mardi à 13-04.m4a',name='mardi à 13-04.m4a',speaker=9112))

labels = np.load(os.getenv("CLASSES"))
ss = load(os.getenv("SCALER"))
model = load_model(np.load(os.getenv("CLASSES")))
data = []
while True:
  content = consumer.read_next()
  data.append(content.value())
  if len(data) >=5 :
    
    current_df = pd.DataFrame.from_records([x.__dict__ for x in data]);
    df = pd.concat([total_df,current_df],ignore_index=True)
    fileNumber = 0
    for fle in glob.glob("feature_label/features_label*.npy"):
      print(fle)
      regex = re.compile(r'\d+')
      x = regex.findall(fle)[0]
      if fileNumber < int(x): 
        fileNumber = int(x) 
      fileNumber = fileNumber+1
    current_features = current_df.apply(extract_features, axis=1)
    np.save('feature_label/features_label'+str(fileNumber), fileNumber)
    features_label = np.concatenate((current_features,total_features_label),axis=0)
    features = []
    for i in range(0, len(features_label)):
        features.append(np.concatenate((features_label[i][0], features_label[i][1], 
                    features_label[i][2], features_label[i][3],
                    features_label[i][4]), axis=0))
    X = np.array(features)
    for y in data:
      if str(y.speaker) not in labels:
        labels = np.append(labels,y.speaker)
    X = ss.transform(X)
    y = np.array(df['speaker']).astype(int)
    lb = LabelEncoder()
    y = to_categorical(lb.fit_transform(y))
    X.shape
    y.shape
    np.save(os.getenv("CLASSES"),lb.classes_)
    X_train = ss.transform(X)    
    model.pop()
    model.add(Dense(len(labels), activation = 'softmax',name='Classes_layer',bias_initializer=initBias))
    layer_names = [layer.name for layer in model.layers]
    layer_idx = layer_names.index('Classes_layer')
    weights[0].resize((128,len(labels)),refcheck=False)
    weights[1].resize((len(labels)),refcheck=False)
    model.layers[layer_idx].set_weights(weights)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    dump(ss, os.getenv("SCALER"), compress=True)
    history = model.fit(X_train, y, batch_size=256, epochs=100, 
                        callbacks=[early_stop])
    model.save(np.load(os.getenv("CLASSES")))
    data = []