# https://thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python
# https://stackoverflow.com/questions/57268372/how-to-convert-live-real-time-audio-from-mic-to-text
import os
import pyaudio
import threading
import speech_recognition as sr


class SpeechToText:

    def __init__(self, save_dir='./test', DEBUG=False):
        self.DEBUG = DEBUG
        self.MIC_IDX = 1

        self.filename = 'test.txt'
        self.save_dir = save_dir

        print('...loading speech-to-text')
        self.r = sr.Recognizer()
        self.p = pyaudio.PyAudio()

        if self.DEBUG:
            self.find_microphone()

    def find_microphone(self):
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))

    def get_speech(self):
        with sr.Microphone(self.MIC_IDX) as source:
            self.r.adjust_for_ambient_noise(source, 1)
            print('Please speak into the mic...')
            audio = self.r.listen(source)
            print('...converting speech-to-text')
            
            try:
                text = self.r.recognize_google(audio)
                if self.DEBUG:
                    print(text)
                return text
            except sr.exceptions.UnknownValueError as e:
                print('Could not understand, try speaking more clearly')
                return -1
            return -1

    def record_speech(self):
        text = self.get_speech()
        if text != -1:
            file_path = self.save_dir + '/' + self.filename

            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)

            with open(file_path, "w") as f:
                f.write(text)

    def start(self, filename):
        self.filename=filename
        #self.record_speech()
        t = threading.Thread(target=self.record_speech)
        t.start()


if __name__ == "__main__":
    speech_to_text = SpeechToText(DEBUG=True)
    text = speech_to_text.get_speech()
    print(text)
