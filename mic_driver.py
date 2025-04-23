import os
import pyaudio
import threading
import speech_recognition as sr
import sys
import ctypes

# Suppress ALSA warnings
def suppress_alsa_errors():
    try:
        asound = ctypes.cdll.LoadLibrary('libasound.so')
        asound.snd_lib_error_set_handler(None)
    except:
        pass

suppress_alsa_errors()

class SpeechToText:

    def __init__(self, save_dir='./test', DEBUG=False):
        self.DEBUG = DEBUG
        self.MIC_IDX = 1
        self.filename = 'test.txt'
        self.save_dir = save_dir

        print('...loading speech-to-text')
        self.r = sr.Recognizer()
        self.p = pyaudio.PyAudio()
        self.r.energy_threshold = 100  # Lower for sensitivity

        if self.DEBUG:
            self.find_microphone()

    def find_microphone(self):
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))

    def get_speech(self):
        with sr.Microphone(self.MIC_IDX) as source:
            print('Please speak into the mic...')
            self.r.adjust_for_ambient_noise(source, duration=1)
            audio = self.r.listen(source, timeout=None, phrase_time_limit=10)
            print('...converting speech-to-text')

            try:
                text = self.r.recognize_google(audio)
                print(f"Speech understood: \"{text}\"")
                return text
            except sr.UnknownValueError:
                print('Could not understand, try speaking more clearly')
                return -1
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return -1

    def record_speech(self):
        text = -1
        while text == -1:
            text = self.get_speech()
        file_path = os.path.join(self.save_dir, self.filename)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        with open(file_path, "w") as f:
            f.write(text)

    def start(self, filename):
        self.filename = filename
        self.thread = threading.Thread(target=self.record_speech)
        self.thread.start()

if __name__ == "__main__":
    speech_to_text = SpeechToText(DEBUG=True)
    text = speech_to_text.get_speech()

