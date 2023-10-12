#!/usr/bin/env python
# coding: utf-8

# # Workflow:
# 
# 1. Importing necessary packages
# 2. Feature exraction and creation of dataset
# 3. Train - test split
# 4. Feature scaling
# 5. Model building
# 6. Evaluation metrics
# 7. Random value prediction
# 8. Saving the model

# ## Importing necessary packages

# In[2]:


import os
import numpy as np
from tensorflow import keras
import librosa
from matplotlib import pyplot
import flask
import pandas as pd
import soundfile as sf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV
#from keras.wrappers.scikit_learn import KerasClassifier
import pickle


# ## Feature extraction using librosa

# 1. Mel frequency Cepstral coefficients
# 2. Mel spectrogram
# 3. Chroma vector
# 4. Tonal Centroid Features

# ### Mel frequency cepstral coefficients  

# In[3]:


#Defining a function that passses audio file as a parameter to obtain MFCC
def get_mfcc(file_path):
    
    #Converting the mp3 audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return MFCC
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    mfcc=np.array(librosa.feature.mfcc(y=y,sr=sr))
    return mfcc


# In[4]:



# ### Mel Spectrogram 

# In[5]:


def get_melspectrogram(file_path):
    
    #Converting the mp3 audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return melspectrogram
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    melspectrogram=np.array(librosa.feature.melspectrogram(y=y,sr=sr))
    return melspectrogram

# ### Chroma vector 

# In[7]:


def get_chroma_vector(file_path):
    #Converting the audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return chroma
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    chroma=np.array(librosa.feature.chroma_stft(y=y,sr=sr))
    return chroma


# In[8]:



# ### Tonal centroid features 

# In[9]:


def get_tonnetz(file_path):
    #Converting the audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return tonnetz
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    tonnetz=np.array(librosa.feature.tonnetz(y=y,sr=sr))
    return tonnetz


#

# In[11]:


#Defining a function to extract all features
def get_features(file_path):
    
    #Converting the audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #MFCC
    mfcc=get_mfcc("temp.wav")
    mfcc_feature=np.concatenate((mfcc.mean(axis=1),mfcc.min(axis=1),mfcc.max(axis=1)))
    
    #Melspectrogram
    melspectrogram=get_melspectrogram("temp.wav")
    melspectrogram_feature=np.concatenate((melspectrogram.mean(axis=1),melspectrogram.min(axis=1),melspectrogram.max(axis=1)))
    
    #Chroma
    chroma=get_chroma_vector("temp.wav")
    chroma_feature=np.concatenate((chroma.mean(axis=1),chroma.min(axis=1),chroma.max(axis=1)))
    
    #Tonnetz
    tntz=get_tonnetz("temp.wav")
    tntz_feature=np.concatenate((tntz.mean(axis=1),tntz.min(axis=1),tntz.max(axis=1)))
    
    #All features
    feature=np.concatenate((chroma_feature,melspectrogram_feature,mfcc_feature,tntz_feature))
    return feature


# ## Calculating features for the full dataset 

# In[12]:


#Defining directory, genres, feature and labels
directory="/Users/casarulez/Projects/Genre_classification/genres"
genres=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
features=[]
labels=[]

#Iterating over data directory to calculate features
for genre in genres:
    for file in os.listdir(directory+"/"+genre):
            file_path=directory+"/"+genre+"/"+file
            
            features.append(get_features(file_path))
            label=genres.index(genre)
            labels.append(label)


# In[13]:


#Shape of features and labels list
print(len(features))
print(len(labels))


# ## Splitting the dataset into training, validation and testing 

# In[14]:


#Shufling features and labels
permutations=np.random.permutation(1000)
features=np.array(features)[permutations]
labels=np.array(labels)[permutations]

#Training data
features_train=features[0:600]
labels_train=labels[0:600]

#Validation data
features_val=features[600:800]
labels_val=labels[600:800]

#Testing data
features_test=features[800:1000]
labels_test=labels[800:1000]

#Checking shape of training and testing variables
features_train.shape,features_test.shape,labels_train.shape,labels_test.shape


# ## Feature scaling 

# In[15]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[16]:


#Transforming features_train to a standard scale
features_train=sc.fit_transform(features_train)
features_train


# In[17]:


features_train.tolist()
print(len(features_train))


# In[18]:


#Transforming features_test to a standard scale
features_test=sc.fit_transform(features_test)
features_test


# In[19]:


features_test.tolist()
print(len(features_test))


# ## Model building 

# In[20]:


#Adding input layer, 2 dense layers and output later
inputs=keras.Input(shape=(498), name="feature")
x=keras.layers.Dense(300,activation="tanh",name="dense_1")(inputs)
x=keras.layers.Dense(200,activation="relu",name="dense_2")(x)
outputs=keras.layers.Dense(10,activation="softmax",name="predictions")(x)

#Defining model
model=keras.Model(inputs=inputs,outputs=outputs)

#Compiling model
model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy()])


#Fitting the model
model.fit(features_train.tolist(),labels_train.tolist(),verbose=1,epochs=100)


# ## Saving the model 

pickle.dump(model,open('model.pkl','wb'))

model_load=pickle.load(open('model.pkl','rb'))

# In[ ]:




