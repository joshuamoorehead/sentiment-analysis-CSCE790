import os
import time
import threading
from mic_driver import SpeechToText
from cam_driver import CaptureVideo

# init
save_dir = './tmp'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

#speech_to_text = SpeechToText(save_dir)

# run
# TODO: recording must happen at the same time

# TODO: get image from camera
t1 = CaptureVideo()
t1.save_dir = save_dir

# get speech from microphone
# TODO: loop until a good audio signal is received
t2 = SpeechToText()
t2.save_dir = save_dir

print('readying recording devices..')
time.sleep(2)

filename = str(time.time())
#print(dir(t2.start()))
t1.start(filename + ".png")
t2.start(filename + ".txt")

# TODO: run multimodal expresion recognition
