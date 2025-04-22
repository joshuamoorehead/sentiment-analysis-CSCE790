import os
import time
import threading
from mic_driver import SpeechToText
from cam_driver import CaptureVideo
from run_sentiment_analysis import SentimentAnalysis
#from multimodalfusion import late_fusion_model, preprocessing

# init
save_dir = './tmp'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# run
# gets image from camera
t1 = CaptureVideo()
t1.save_dir = save_dir

# get speech from microphone
# TODO: loop until a good audio signal is received
t2 = SpeechToText()
t2.save_dir = save_dir

print('readying recording devices..')
time.sleep(2)

filename = str(time.time())
#t1.start(filename + ".png")
t2.start(filename + ".txt")

# TODO fer
#output = FER(video)

# sentiment analysis
# load model
sentimentanalysis = SentimentAnalysis(True)
bert = sentimentanalysis.load_model('./weights/model.pt')

# inference on output
with open(save_dir + '/' + filename + '.txt', 'r') as f:
    text = f.read()
sentiment_prediction = sentimentanalysis.inference(bert, text)

# TODO: run multimodal expresion recognition
final = late_fusion_model(output, sentiment_prediction)
print('final prediction:', final)
