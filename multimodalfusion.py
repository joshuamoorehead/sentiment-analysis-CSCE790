# multimodalfusion.py
import math
import numpy as np
import torch
import torch.nn as nn
import sys

# Emotion mappings
fer_classes = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
bert_classes = ['negative', 'positive']  # 0 or 1

'''
Providing 'soft' probabilities for BERT-sentiment-FER outputs
BERT outputs {0, 1} (2 classes) | FER outputs {anger ... surprise} (7 classes)
'''
tone_to_face = {
    1: {  # positive BERT sentiment
        'happiness': 1.0,
        'surprise': 0.8,
        'neutral': 0.6,
        'sadness': 0.3,
        'anger': 0.2,
        'disgust': 0.1,
        'fear': 0.2
    },
    0: {  # negative BERT sentiment
        'happiness': 0.2,
        'surprise': 0.4,
        'neutral': 0.5,
        'sadness': 1.0,
        'anger': 0.8,
        'disgust': 0.7,
        'fear': 0.6
    }
}

# Late fusion model script
class late_fusion_model(nn.Module):
    def __init__(self):
        super(late_fusion_model, self).__init__()
        # FER has 7 classes, BERT has 2 classes, so we have 9 input dimensions
        self.fc = nn.Linear(9, 3)  # Output: negative, positive, neutral/conflicting
        self.softmax = nn.Softmax(dim=1)  # Convert to probability distribution
    
    def forward(self, bert_pred, fer_pred):
        '''
        BERT: tensor [1, 2] - probabilities for [negative, positive]
        FER: tensor [1, 7] - probabilities for 7 emotions
        '''
        # Get the dominant sentiment class from BERT (0=negative, 1=positive)
        sentiment_class = int(torch.argmax(bert_pred, dim=1).item())
        
        # Get the weighting map for the detected sentiment
        class_from_map = tone_to_face[sentiment_class]
        
        # Apply weights to FER predictions based on BERT sentiment
        weighted_fer = []
        for i, cls in enumerate(fer_classes):
            weighted_fer.append(fer_pred[0, i].item() * class_from_map[cls])
        
        # Convert to tensor
        weighted_fer = torch.tensor([weighted_fer], dtype=fer_pred.dtype, device=fer_pred.device)
        
        # Concatenate BERT and weighted FER for final classification
        combined_input = torch.cat([bert_pred, weighted_fer], dim=1)  # [1, 9]
        
        # Final classification layer
        sentiment = self.fc(combined_input)
        sentiment = self.softmax(sentiment)
        
        return sentiment

def predict_emotion(bert_output, fer_output):
    """
    Combine BERT and FER outputs to predict final emotion
    
    Args:
        bert_output: Dict with sentiment (0/1) and probabilities
        fer_output: Dict with emotion, sentiment, confidence, probabilities
    
    Returns:
        dict: Prediction with emotion, confidence, etc.
    """
    # Create model
    model = late_fusion_model()
    
    # Extract probabilities
    bert_probs = bert_output.get("probabilities", [0.5, 0.5])
    fer_probs = fer_output.get("probabilities", [0.0] * 7)
    
    # Convert to tensors
    bert_tensor = torch.tensor([bert_probs])
    fer_tensor = torch.tensor([fer_probs])
    
    # Get fusion prediction
    with torch.no_grad():
        fusion_output = model(bert_tensor, fer_tensor)
        prediction = torch.argmax(fusion_output, dim=1).item()
        confidence = fusion_output[0, prediction].item()
    
    # Map prediction to sentiment
    sentiment_map = {
        0: "negative",
        1: "positive",
        2: "neutral/conflicting"
    }
    
    sentiment = sentiment_map[prediction]
    
    return {
        "emotion": sentiment,
        "confidence": confidence,
        "probabilities": fusion_output[0].tolist()
    }