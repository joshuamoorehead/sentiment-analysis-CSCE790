import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from train.patt_lite import PAttLite

# Load model
MODEL_PATH = "models/patt_lite.weights.h5"
IMG_SHAPE = (120, 120, 3)

# Emotion mapping
EMOTION_MAP = {
    0: "anger",     # negative
    1: "disgust",   # negative
    2: "fear",      # negative
    3: "happiness", # positive
    4: "neutral",   # neutral
    5: "sadness",   # negative
    6: "surprise"   # neutral
}

SENTIMENT_MAP = {
    "anger": -1,
    "disgust": -1,
    "fear": -1,
    "happiness": 1,
    "neutral": 0,
    "sadness": -1,
    "surprise": 0
}

# Load model
_model = PAttLite()
_model.build((None, *IMG_SHAPE))
_model.load_weights(MODEL_PATH)

def predict_emotion_from_path(image_path):
    img = cv2.imread(image_path)
    return predict_emotion_from_array(img)

def predict_emotion_from_array(image):
    if image.shape[:2] != IMG_SHAPE[:2]:
        image = cv2.resize(image, IMG_SHAPE[:2])
    image = np.expand_dims(image, axis=0)

    preds = _model(image, training=False).numpy()
    emotion_idx = np.argmax(preds[0])
    emotion = EMOTION_MAP[emotion_idx]
    sentiment = SENTIMENT_MAP[emotion]
    confidence = preds[0][emotion_idx]

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "confidence": float(confidence)
    }
