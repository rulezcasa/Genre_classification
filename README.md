# Music genre classification - A machine learning tool

## Description:
This tool aims to classify and predict the genre of input music into 10 genres:
 1. Blues
 2. Classical
 3. Country
 4. Disco
 5. Hiphop
 6. Jazz
 7. Metal
 8. Pop
 9. Reggae
 10. Rock

## Installations:
The project uses the following python packages and dependencies for its successful compilation:
1. numpy
2. pandas
3. os
4. librosa
5. soundfile
6. matplotlib
7. tensorflow.keras

## Dataset:
GTZAN Genre Collection: https://www.kaggle.com/datasets/carlthome/gtzan-genre-collection

The dataset contains contains directories for 10 different genres having 100 audio files each.

## Feature extraction from audio files using librosa: 
The model is trained using certain audio features extracted by the librosa library:
1. Mel frequency capstral coefficients -> MFCCs are a set of coefficients that represent the short-term power spectrum of a sound signal. They are widely used in audio signal processing and speech and music analysis. MFCCs capture essential spectral characteristics of audio signals by mimicking the human auditory system's sensitivity to different frequencies.
2. Mel spectrogram -> A Mel spectrogram is a two-dimensional representation of an audio signal that shows how the signal's power is distributed across time and Mel-frequency bands.
3. Chroma vector -> A chroma vector, also known as a chromagram, is a feature representation used in music and audio signal processing to describe the distribution of musical pitch classes within a segment of audio. It is a valuable tool for tasks such as music analysis, chord recognition, and music genre classification.
4. Tonal centroid features provide a time-varying representation of the harmonic content of the audio signal, which is valuable for various music analysis tasks.

## Workflow followed:
1. Importing necessary packages
2. Feature exraction and creation of dataset
3. Train - test split
4. Feature scaling
5. Model building
6. Evaluation metrics
7. Random value prediction
8. Saving the model

