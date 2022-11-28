######## IMPORTS ##########
import os
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# #### LOADING THE VOICE DATA FOR VISUALIZATION ###
# walley_sample = "audio_data/10_59_121.wav"
# data, fs = librosa.load(walley_sample)
# # seconds = librosa.get_duration(walley_sample)

# ####### ALL CONSTANTS #####
# fs = 44100
# seconds = 3
# filename = "prediction.wav"
# class_names = ["Wake Word NOT Detected", "Wake Word Detected"]

##### LOADING OUR SAVED MODEL and PREDICTING ###
model = load_model("saved_model/Sri_WWD_20221128_200epochs.h5")

all_data = []

data_path_dict = {
    0: ["background_sound/" + file_path for file_path in os.listdir("background_sound/")],
    1: ["audio_data/" + file_path for file_path in os.listdir("audio_data/")],
    2: ["error_data/" + file_path for file_path in os.listdir("error_data/")]
}

# the background_sound/ directory has all sounds which DOES NOT CONTAIN wake word
# the audio_data/ directory has all sound WHICH HAS Wake word
i = 0
print("Prediction Started: ")
for class_label, list_of_files in data_path_dict.items():
    for single_file in list_of_files:
        audio, sample_rate = librosa.load(single_file) ## Loading file
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) ## Apllying mfcc
        mfcc_processed = np.mean(mfcc.T, axis=0) ## some pre-processing
        print(single_file)
        print(sample_rate)
        # print(mfcc_processed)
        print(class_label)

        prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
        type(prediction)
        p= np.round(prediction).flatten()
        print("rounded predication value :",p)
        # print(prediction)
       # if prediction[:, 1] > 0.99:
        if p[1] == 1:
            print(f"Wake Word Detected for ({i})")
            print("Confidence:", prediction[:, 1])
            i += 1
        elif p[2] == 1:
            print(f"Error Detected for ({i})")
            print("Confidence:", prediction[:, 1])
         
        else:
            print(f"Wake Word NOT Detected")
            print("Confidence:", prediction[:, 0])

    print(f"Info: Succesfully Preprocessed Class Label {class_label}")



# i = 0
# while True:
#     print("Say Now: ")
#     myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
#     #myrecording = sd.rec(int(seconds * fs), samplerate=fs,channels=2)
#     sd.wait()
#     sd.play(myrecording)
#     sd.wait()
#     write(filename, fs, myrecording)

#     audio, sample_rate = librosa.load(filename)
#     mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
#     mfcc_processed = np.mean(mfcc.T, axis=0)

#     prediction = model.predict(np.expand_dims(mfcc_processed, axis=0))
#     if prediction[:, 1] > 0.99:
#         print(f"Wake Word Detected for ({i})")
#         print("Confidence:", prediction[:, 1])
#         i += 1
    
#     else:
#         print(f"Wake Word NOT Detected")
#         print("Confidence:", prediction[:, 0])

#     sd.play(myrecording)
