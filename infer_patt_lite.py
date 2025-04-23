import numpy as np
import cv2
import os
import time
import tensorflow as tf
from models.patt_lite import PAttLite
import torch  # Needed for the fusion model

# Model configuration
H5_MODEL_PATH = "weights/patt_lite.weights.h5"
IMG_SHAPE = (120, 120, 3)

# Emotion mapping
EMOTION_MAP = {
    0: "anger", 1: "disgust", 2: "fear",
    3: "happiness", 4: "neutral", 5: "sadness", 6: "surprise"
}

SENTIMENT_MAP = {
    "anger": -1, "disgust": -1, "fear": -1,
    "happiness": 1, "neutral": 0, "sadness": -1, "surprise": 0
}

# Load TensorFlow model
_model = None

def _ensure_model_loaded():
    global _model
    if _model is None:
        print(f"Loading model from {H5_MODEL_PATH}")
        _model = PAttLite()
        _model.build((None, *IMG_SHAPE))
        _model.load_weights(H5_MODEL_PATH)
        print("Model loaded successfully")

# Predict emotion from image array
def predict_emotion_from_array(image):
    _ensure_model_loaded()

    if image.shape[:2] != IMG_SHAPE[:2]:
        image = cv2.resize(image, IMG_SHAPE[:2])

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    image = image.astype(np.float32)

    predictions = _model(image, training=False)
    emotion_idx = tf.argmax(predictions[0]).numpy()
    confidence = predictions[0][emotion_idx].numpy()

    emotion = EMOTION_MAP[emotion_idx]
    sentiment = SENTIMENT_MAP[emotion]

    return {
        "emotion": emotion,
        "sentiment": sentiment,
        "confidence": float(confidence),
        "probabilities": predictions[0].numpy().tolist()
    }

def predict_emotion_from_path(image_path):
    """Predict emotion from an image file path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return predict_emotion_from_array(img)

# Get raw probabilities for multimodal fusion
def get_probabilities(image):
    _ensure_model_loaded()

    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from {image}")
    else:
        img = image.copy()

    if img.shape[:2] != IMG_SHAPE[:2]:
        img = cv2.resize(img, IMG_SHAPE[:2])

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    img = img.astype(np.float32)

    predictions = _model(img, training=False)

    # Convert predictions to torch tensor for fusion model
    return torch.tensor(predictions.numpy())

# Test
def test():
    print("Testing FER model with .h5 weights...")

    temp_dir = "./tmp"
    test_images = [f for f in os.listdir(temp_dir) if f.endswith('.jpg')]
    if not test_images:
        print("No test images found.")
        return False

    test_image_path = os.path.join(temp_dir, test_images[0])
    print(f"Using test image: {test_image_path}")

    img = cv2.imread(test_image_path)
    result = predict_emotion_from_array(img)

    print("\n=== Results ===")
    print(f"Detected emotion: {result['emotion']}")
    print(f"Sentiment: {result['sentiment']} (-1=negative, 0=neutral, 1=positive)")
    print(f"Confidence: {result['confidence']:.4f}")

    print("\nProbabilities:")
    for i, emotion in EMOTION_MAP.items():
        print(f"  {emotion}: {result['probabilities'][i]:.4f}")

    # Test get_probabilities
    fer_probs = get_probabilities(img)
    print("\nRaw Tensor (for fusion):")
    print(fer_probs)

    return True

if __name__ == "__main__":
    test()

