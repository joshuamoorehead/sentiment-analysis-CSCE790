import os
import time
import threading
import torch
from mic_driver import SpeechToText
from cam_driver import CaptureVideo
from run_sentiment_analysis import SentimentAnalysis
from multimodalfusion import late_fusion_model
from infer_patt_lite import predict_emotion_from_path, get_probabilities

# Init
save_dir = './tmp'
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Camera and Mic setup
t1 = CaptureVideo()
t1.save_dir = save_dir
t2 = SpeechToText()
t2.save_dir = save_dir

print('readying recording devices..')
time.sleep(2)
filename = str(time.time())
t1.start(filename + ".png")
t2.start(filename + ".txt")

# Wait for both threads to complete
t1.thread.join()
t2.thread.join()
print("Capture Completed.")

# Define file paths
img_filename = "0_" + filename + ".png"
image_path = os.path.join(save_dir + "/vid/", img_filename)
text_file = os.path.join(save_dir, filename + '.txt')

print(f"Analyzing image: {image_path}")
print(f"Analyzing text: {text_file}")

# === Facial Expression Analysis ===
fer_result = None
if os.path.exists(image_path):
    print("\n=== Facial Expression Analysis ===")
    fer_result = predict_emotion_from_path(image_path)
    print(f"Detected emotion: {fer_result['emotion']}")
    print(f"Confidence: {fer_result['confidence']:.2f}")

    fer_probs = get_probabilities(image_path)
else:
    print("Could not analyze facial expression: image file not found.")

# === Sentiment Analysis (Text) ===
bert_out = None
if os.path.exists(text_file):
    print("\n=== Sentiment Analysis ===")
    sentimentanalysis = SentimentAnalysis(True)
    bert_model = sentimentanalysis.load_model('./weights/best_model.pt')

    with open(text_file, 'r') as f:
        text = f.read()
    print(f"Text content: {text}")

    sentiment_prediction = sentimentanalysis.inference(bert_model, text)
    sentiment_label = "positive" if sentiment_prediction.item() == 1 else "negative"
    sentiment_value = 1 if sentiment_prediction.item() == 1 else -1
    print(f"Detected sentiment: {sentiment_label} ({sentiment_value})")

    bert_out = torch.zeros((1, 2))
    bert_out[0, 0 if sentiment_prediction == 1 else 1] = 1.0
else:
    print("Could not analyze sentiment: text file not found.")

# === Multimodal Fusion ===
if fer_result is not None and bert_out is not None:
    print("\n=== Running Multimodal Fusion ===")

    model = late_fusion_model()
    final_sentiment = model(bert_out, fer_probs)

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment_value_map = {0: -1, 1: 0, 2: 1}

    final_idx = torch.argmax(final_sentiment, dim=1).item()

    print("\n=== Final Multimodal Result ===")
    print(f"Combined sentiment: {sentiment_map[final_idx]}")
    print(f"Sentiment value: {sentiment_value_map[final_idx]} (-1=negative, 0=neutral, 1=positive)")
    print(f"Confidence: {final_sentiment[0, final_idx].item():.2f}")
else:
    print("\nMultimodal fusion skipped: missing FER or BERT results")

