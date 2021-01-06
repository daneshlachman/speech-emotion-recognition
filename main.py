import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pdb

# path to audio files 
path_to_audio_files = "speech-emotion-recognition-ravdess-data"

# do audio processing so that features (mfcc, chroma, mel) are extracted
#  from the sound file, and then process them into result
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        opened_soundfile = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        # apply fourier transform on opened soundfile
        stft = np.abs(librosa.stft(opened_soundfile))
        result = np.array([])   
        
        
        # Mel-frequency cepstral coefficients (MFCCs) 
        mfccs = np.mean(librosa.feature.mfcc(y=opened_soundfile, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        pdb.set_trace()
        # Compute a chromagram from a waveform or power spectrogram.
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
        pdb.set_trace()
        # Compute a mel-scaled spectrogram.
        mel = np.mean(librosa.feature.melspectrogram(opened_soundfile, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
        pdb.set_trace()
    return result


#Load the data and extract features and emotions for each sound file, 
# then create and return a train_test_split for this data
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob( path_to_audio_files + "\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

# the to be observed emotions
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')