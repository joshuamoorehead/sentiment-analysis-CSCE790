import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

class PAttLite(tf.keras.Model):
    def __init__(self, num_classes=7, img_shape=(120, 120, 3), dropout_rate=0.1):
        super(PAttLite, self).__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        
        # Define layers
        self.sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode='horizontal'), 
            tf.keras.layers.RandomContrast(factor=0.3)
        ], name="augmentation")
        self.preprocess_input = tf.keras.applications.mobilenet.preprocess_input
        
        # Backbone
        backbone = tf.keras.applications.mobilenet.MobileNet(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        backbone.trainable = False
        self.base_model = tf.keras.Model(
            backbone.input, backbone.layers[-29].output, name='base_model')
        
        # Patch extraction
        self.patch_extraction = tf.keras.Sequential([
            tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'), 
            tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'), 
            tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu')
        ], name='patch_extraction')
        
        # Attention and classification
        self.self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.pre_classification = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'), 
            tf.keras.layers.BatchNormalization()
        ], name='pre_classification')
        self.prediction_layer = tf.keras.layers.Dense(num_classes, activation="softmax", name='classification_head')
    
    def call(self, inputs, training=False):
        x = self.sample_resizing(inputs)
        if training:
            x = self.data_augmentation(x)
        x = self.preprocess_input(x)
        x = self.base_model(x, training=False)
        x = self.patch_extraction(x)
        x = self.global_average_layer(x)
        x = self.dropout(x)
        x = self.pre_classification(x)
        x = self.self_attention([x, x])
        outputs = self.prediction_layer(x)
        return outputs
    
    def predict_emotion(self, image):
        """Convert emotion prediction to binary positive/negative"""
        # Ensure image has correct shape
        if image.shape[:2] != self.img_shape[:2]:
            image = cv2.resize(image, self.img_shape[:2])
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get prediction
        predictions = self(image, training=False)
        
        # Get emotion index and confidence
        emotion_idx = tf.argmax(predictions[0]).numpy()
        confidence = predictions[0][emotion_idx].numpy()
        
        # Map to positive/negative sentiment
        # Assuming: 0=anger, 1=disgust, 2=fear, 3=happiness, 4=neutral, 5=sadness, 6=surprise
        emotion_map = {
            0: "anger",     # negative
            1: "disgust",   # negative
            2: "fear",      # negative
            3: "happiness", # positive
            4: "neutral",   # neutral
            5: "sadness",   # negative
            6: "surprise"   # neutral
        }
        
        sentiment_map = {
            "anger": -1,
            "disgust": -1,
            "fear": -1,
            "happiness": 1,
            "neutral": 0,
            "sadness": -1,
            "surprise": 0
        }
        
        emotion = emotion_map[emotion_idx]
        sentiment = sentiment_map[emotion]
        
        return emotion, sentiment, confidence
    
    def save_weights_to_file(self, filepath):
        self.save_weights(filepath)
    
    def load_weights_from_file(self, filepath):
        self.load_weights(filepath)