import os
import numpy as np
from tensorflow import keras
import librosa
from matplotlib import pyplot
import flask
import pandas as pd
import soundfile as sf
import pickle
from flask import Flask,request,jsonify,render_template
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

#Defining a function that passses audio file as a parameter to obtain MFCC
def get_mfcc(file_path):
    
    #Converting the mp3 audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return MFCC
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    mfcc=np.array(librosa.feature.mfcc(y=y,sr=sr))
    return mfcc

def get_melspectrogram(file_path):
    
    #Converting the mp3 audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return melspectrogram
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    melspectrogram=np.array(librosa.feature.melspectrogram(y=y,sr=sr))
    return melspectrogram

def get_chroma_vector(file_path):
    #Converting the audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return chroma
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    chroma=np.array(librosa.feature.chroma_stft(y=y,sr=sr))
    return chroma

def get_tonnetz(file_path):
    #Converting the audio data to wav as supported by librosa
    data, samplerate = sf.read(file_path)
    sf.write("temp.wav", data, samplerate)
    
    #Passing the audio file to return tonnetz
    y,sr=librosa.load("temp.wav",offset=0,duration=30)
    tonnetz=np.array(librosa.feature.tonnetz(y=y,sr=sr))
    return tonnetz

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

genres=["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
    

app=Flask(__name__)
model = keras.models.load_model("genre.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_genre():
  
    audio_file = request.files['audio']
    if audio_file:
       # Define the directory where you want to save the uploaded audio files
       upload_folder = 'uploads'

       # Ensure the 'uploads' directory exists
       os.makedirs(upload_folder, exist_ok=True)

       # Save the uploaded audio file to the specified directory
       audio_file.save(os.path.join(upload_folder, audio_file.filename))
       temp_path=os.path.join(upload_folder,audio_file.filename)
       feat=get_features(temp_path)
       feat=feat.reshape(1,498)
       feat=sc.fit_transform(feat)
       prediction=model.predict(feat)
       ind=np.argmax(prediction)
       output=(genres[ind])
       
       return render_template('index.html', prediction_text='Predicted genre is {}' .format(output) )
   
if __name__ == '__main__':
    app.run(debug=True)
   