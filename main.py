import os
import time
import threading
from mic_driver import SpeechToText
from cam_driver import CaptureVideo
from run_sentiment_analysis import SentimentAnalysis
#from multimodalfusion import late_fusion_model, preprocessing
#from infer_patt_lite import predict_emotion_from_path

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

# Wait for both threads to complete
#t1.join()
#t2.join()
#print("Capture Completed.")

# TODO fer
"""
# Facial Expression
img_filename = "0_" + filename + ".png"
image_path = os.path.join(save_dir + "/vid/", img_filename)

if os.path.exists(image_path):
    print("Analyzing facial expression...")
    fer_result = predict_emotion_from_path(image_path)
    print("\n=== Facial Expression Analysis ===")
    print(f"Detected emotion: {fer_result['emotion']}")
    print(f"Sentiment: {fer_result['sentiment']} (-1=negative, 0=neutral, 1=positive)")
    print(f"Confidence: {fer_result['confidence']:.2f}")
else:
    print("Could not analyze facial expression: image file not found.")
"""
# sentiment analysis
# load model
sentimentanalysis = SentimentAnalysis(True)
bert = sentimentanalysis.load_model('./weights/best_model.pt')

# inference on output
with open(save_dir + '/' + filename + '.txt', 'r') as f:
    text = f.read()
sentiment_prediction = sentimentanalysis.inference(bert, text)

# TODO: run multimodal expresion recognition
#final = late_fusion_model(output, sentiment_prediction)
#print('final prediction:', final)
