import threading
from mic_driver import SpeechToText

# init
speech_to_text = SpeechToText()

# run
# TODO: recording must happen at the same time

# TODO: get image from camera

# get speech from microphone
# TODO: loop until a good audio signal is received
text = threading.Thread(target=speech_to_text.get_speech())


# TODO: run multimodal expresion recognition
