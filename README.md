# Speech Emotion Recognizer
This project will be able to predict emotions from human speech. The data for this project comes from [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.X_X8EdhKiHs).
This dataset has 7356 files rated by 247 individuals 10 times on emotional validity, intensity, and genuineness. However, for the sake of simplicity, the used files have a lowered sample rate in this project.

For the audio processing part, the libraries librosa and soundfile have been used. For the calculations/machine learning part, the common libraries numpy and sckikit-learn have been used. For the learning part, the Multi-layer Perceptron Classifier was used.

The accuracy of the model is, roughly, 65%-75%.

