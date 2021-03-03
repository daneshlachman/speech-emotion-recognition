import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self):
        self.model = None

        # path to audio files & prediction testfile
        self.path_to_audio_files = "../data/speech-emotion-recognition-ravdess-data"
        self.prediction_file = "../data/my_dad_pretending_to_be_sad.wav"

        # emotions in the RAVDESS dataset
        self.observed_emotions = ['calm', 'happy', 'fearful', 'disgust']
        self.emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }

    def define_model(self):
        # Split the dataset, and print some information about the data
        x_train, x_test, y_train, y_test = self.load_data(test_size=0.25)
        print(f'Features extracted: {x_train.shape[1]}')

        # # train the model and make prediction
        self.model = MLPClassifier(activation="relu", early_stopping=False, shuffle=True,  alpha=0.01, batch_size=256,
                                   epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
        self.model.fit(x_train, y_train)
        y_predict = self.model.predict(x_test)

        # calculate & print the accuracy of our model
        accuracy = accuracy_score(y_true=y_test, y_pred=y_predict)
        print("The accuracy of the model is: {:.2f}%".format(accuracy*100))
        print("The predicted emotion for the given sample file is: " + self.make_prediction(self.prediction_file)[0])

    # Load the data and extract features and emotions for each sound file,
    # then create and return a train_test_split for this data
    def load_data(self, test_size=0.2):
        x, y = [], []
        for file in glob.glob(self.path_to_audio_files + "\\Actor_*\\*.wav"):
            file_name = os.path.basename(file)
            emotion = self.emotions[file_name.split("-")[2]]
            if emotion not in self.observed_emotions:
                continue
            feature = self.extract_feature(file)
            x.append(feature)
            y.append(emotion)
        return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

    def make_prediction(self, file_name):
        sample_feature = self.extract_feature(file_name)
        reshaped_sample_feature = sample_feature.reshape(1, -1)
        prediction = self.model.predict(reshaped_sample_feature)
        return prediction

    @staticmethod
    # do audio processing so that features (mfcc, chroma, mel) are extracted
    # from the sound file, and then process them into result for later learning
    def extract_feature(file_name):
        with soundfile.SoundFile(file_name) as sound_file:
            opened_soundfile = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            # apply fourier transform on opened soundfile
            stft = np.abs(librosa.stft(opened_soundfile))
            result = np.array([])

            # Mel-frequency cepstral coefficients (MFCCs)
            mfccs = np.mean(librosa.feature.mfcc(y=opened_soundfile, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

            # Compute a chromagram from a waveform or power spectrogram.
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

            # Compute a mel-scaled spectrogram.
            mel = np.mean(librosa.feature.melspectrogram(opened_soundfile, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))

        return result

