import librosa
import librosa.display
import soundfile
import os, glob, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
        
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


hesitations={
 '00':'no',
  '01':'yes'}

#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[] # x = features, y = emotions
    for file in glob.glob('/Users/MihyeC/Desktop/hesitation_det/dataset/sorted-speech-silence/Actor_*/*.wav'):
        file_name=os.path.basename(file)
        hesitation=hesitations[file_name.split("-")[1]]
        feature=extract_feature(file, mfcc=True, chroma=True, tempo=True)
        x.append(feature)
        y.append(hesitation)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train,x_test,y_train,y_test=load_data(test_size=0.2)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

x_train = x_train.astype(int)
x_test = x_test.astype(int)

model.fit(x_train,y_train)
y_pred_training = model.predict(x_train)
y_pred=model.predict(x_test)

accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_train,y_pred_training))
print(classification_report(y_train,y_pred_training))

#save the model
filename = 'mlpmodel.sav'
pickle.dump(model, open(filename, 'wb'))
