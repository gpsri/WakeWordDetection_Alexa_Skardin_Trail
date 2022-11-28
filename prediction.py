######## IMPORTS ##########
import os
import time
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

#### LOADING THE VOICE DATA FOR VISUALIZATION ###
walley_sample = "audio_data/10_59_1215.wav"
data, fs = librosa.load(walley_sample)
# seconds = librosa.get_duration(walley_sample)

####### ALL CONSTANTS #####
fs = 44100
seconds = 3
filename = "prediction.wav"

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/Sri_WWD_20221128_200epochs.h5")

print("Prediction Started: ")
i = 0
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    #myrecording = sd.rec(int(seconds * fs), samplerate=fs,channels=2)
    sd.wait()
    sd.play(myrecording)
    sd.wait()
    write(filename, fs, myrecording)
    time.sleep(1)

    audio, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_processed = np.mean(mfcc.T, axis=0)

    prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
    p= np.round(prediction).flatten()
    print(type(p))
    print(p[0])
    print(p[1])
    print(p[2])
    print("rounded predication value :",p)
    # if prediction[:, 1] > 0.99:
    if p[1] == 1:
        print(f"Wake Word Detected for ({i})")
        print("Confidence:", prediction[:, 1])
        i += 1
    
    else:
        print(f"Wake Word NOT Detected")
        print("Confidence:", prediction[:, 0])

