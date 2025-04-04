# https://thepythoncode.com/article/using-speech-recognition-to-convert-speech-to-text-python
# https://stackoverflow.com/questions/57268372/how-to-convert-live-real-time-audio-from-mic-to-text
import pyaudio
import speech_recognition as sr


class SpeechToText:

    def __init__(self, DEBUG=False):
        self.DEBUG = DEBUG
        self.MIC_IDX = 2

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


if __name__ == "__main__":
    speech_to_text = SpeechToText()
    text = speech_to_text.get_speech()
    print(text)
