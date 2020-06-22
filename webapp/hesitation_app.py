import os
import io
import os, glob, pickle
import numpy as np
import pandas as pd
import wave
from scipy.io import wavfile
import librosa
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st

st.title("Hesitation Detector")

st.header("Upload your customer file")

AUDIO_EXTENSIONS = ["wav", "flac", "mp3", "aac", "ogg", "oga", "m4a", "opus", "wma"]

# For samples of sounds in different formats, see
# https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/audio-samples.html
## Loading audio 

def get_audio_files_in_dir(directory):
    out = []
    for item in os.listdir(directory):
        try:
            name, ext = item.split(".")
        except:
            continue
        if name and ext:
            if ext in AUDIO_EXTENSIONS:
                out.append(item)
    return out

path= "/Users/MihyeC/Desktop/hesitation_det/dataset"
avdir = os.path.expanduser(path)
audiofiles = get_audio_files_in_dir(avdir)

if len(audiofiles) == 0:
    st.write(
        "Put some audio files in your home directory to activate this player."
        % avdir
    )

else:
    filename = st.selectbox(
        "Select an audio file from your home directory (%s) to play" % avdir,
        audiofiles,
        0,
    )
    audiopath = os.path.join(avdir, filename)
    st.audio(audiopath)

# feature sentences 
def extract_feature(file_name, mfcc, chroma, tempo):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        sample_rate=sample_rate*2
        hop_length=512
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if tempo:
            oenv = librosa.onset.onset_strength(y=X, sr=sample_rate, hop_length=hop_length)
            tempo=np.mean(librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sample_rate, hop_length=hop_length).T,axis=0)
            result=np.hstack((result, tempo))
    return result

import pickle
loaded_model = pickle.load(open('mlpmodel.sav', 'rb'))

## predict new input as hesitation vs. non-hesitation
newinput_feature = extract_feature(audiopath, mfcc=True, chroma=True, tempo=True)
z= []
z.append(newinput_feature)
newdata = np.array(z).astype(int)
y_pred2=loaded_model.predict(newdata)
finalstate=[]
if y_pred2=='no':
    finalstate= "Nope, this customer is not hesitating"
else:
    finalstate="Hmm! this customer is hesitating"

print(y_pred2)
st.write('Is this customer hesitating?')
st.write(finalstate)

