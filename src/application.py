import tkinter as Tkinter
from tkinter import messagebox, Label
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import pyaudio

from model import Model


class Application:
    def __init__(self):
        # initiate model class
        self.model = Model()
        # self.model.define_model()

        # create root window
        self.root = Tkinter.Tk()
        self.root.resizable(False, False)
        self.root.title('Speech Emotion Recognizer')

        self.application_width_and_height = 400
        self.drawing_line_thickness = 17

        # create canvas to display future image on
        self.canvas = Tkinter.Canvas(width=self.application_width_and_height,
                                     height=self.application_width_and_height)
        self.canvas.pack()

        # definition of buttons and their triggered functions
        clear_button = Tkinter.Button(self.root, text="Record", command=self.record_audio_input)
        clear_button.pack()

        predict_button = Tkinter.Button(self.root, text="Predict", command=None)
        predict_button.pack()



        # define labels for displaying predicted digit and its accuracy
        # self.prediction_label = Label(text="Amount of sec to record").place(x=245, y=380)
        # amnt_of_sec_to_rec = Tkinter.Entry(self.root).place(x=250, y=402)
        self.prediction_label = Label(text="Prediction: ").place(x=280, y=422)
        self.accuracy_label = Label(text="Accuracy:").place(x=280, y=437)

        self.root.mainloop()

    @staticmethod
    def record_audio_input():
        chunk = 2048
        rate = 44100
        amount_of_seconds = 4

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                        frames_per_buffer=chunk)

        for i in range(int(rate / chunk * amount_of_seconds)):
            data = np.fromstring(stream.read(chunk), dtype=np.int16)
            peak = np.average(np.abs(data)) * 2
            bars = "#" * int(50 * peak / 2 ** 16)
            print("%04d %05d %s" % (i, peak, bars))

        stream.stop_stream()
        stream.close()
        p.terminate()


Application()



