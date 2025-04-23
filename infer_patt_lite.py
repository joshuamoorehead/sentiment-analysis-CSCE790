import tensorflow as tf
import numpy as np
import cv2
from models.patt_lite import PAttLite  # Fixed import path

# Load model
MODEL_PATH = "weights/patt_lite.weights.h5"
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
_model.build((None, *IMG_SHAPE))  # Fixed syntax
_model.load_weights_from_file(MODEL_PATH)  # Fixed method name

def predict_emotion_from_path(image_path):
    """Predict emotion from an image file path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return predict_emotion_from_array(img)

def predict_emotion_from_array(image):
    """Predict emotion from a numpy array image"""
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
        "confidence": float(confidence),
        "probabilities": preds[0].tolist()
    }

# Function to get raw probabilities for multimodal fusion
def get_probabilities(image):
    """Get raw emotion probabilities for multimodal fusion"""
    if isinstance(image, str):
        # If input is a file path
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        # If input is already a numpy array
        img = image
    
    if img.shape[:2] != IMG_SHAPE[:2]:
        img = cv2.resize(img, IMG_SHAPE[:2])
    
    img = np.expand_dims(img, axis=0)
    preds = _model(img, training=False)
    return preds  # Return raw tensor for fusion model