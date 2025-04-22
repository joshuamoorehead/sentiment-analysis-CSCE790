import os
import time
import threading
from mic_driver import SpeechToText
from cam_driver import CaptureVideo
from infer_patt_lite import predict_emotion_from_path

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
t1.start(filename + ".png")
t2.start(filename + ".txt")

# Wait for capture to complete
print("Capturing data...")
time.sleep(25)  # Camera captures 10 frames with 2 seconds interval

# Get file paths - first frame from camera
img_filename = "0_" + filename + ".png"
image_path = os.path.join(save_dir + "/vid/", img_filename)  # Camera adds /vid/ internally

# Wait for file to exist
max_wait = 10
waited = 0
while not os.path.exists(image_path):
    time.sleep(0.5)
    waited += 0.5
    if waited >= max_wait:
        print(f"Timeout waiting for image file: {image_path}")
        break

# Process the captured image (FER)
if os.path.exists(image_path):
    print("Analyzing facial expression...")
    fer_result = predict_emotion_from_path(image_path)
    
    # Display FER result
    print("\n=== Facial Expression Analysis ===")
    print(f"Detected emotion: {fer_result['emotion']}")
    print(f"Sentiment: {fer_result['sentiment']} (-1=negative, 0=neutral, 1=positive)")
    print(f"Confidence: {fer_result['confidence']:.2f}")
else:
    print("Could not analyze facial expression: image file not found.")

# TODO: run multimodal expresion recognition
