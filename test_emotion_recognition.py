# test_emotion_recognition.py
import os
import cv2
import torch
import numpy as np
from infer_patt_lite import predict_emotion_from_path, predict_emotion_from_array
from multimodalfusion1 import late_fusion_model  # Match your import name

# Mock BERT output for testing
def mock_bert_sentiment(text):
    """Simple rule-based sentiment for testing"""
    positive_words = ["happy", "good", "great", "excellent", "love", "joy"]
    negative_words = ["sad", "bad", "terrible", "hate", "dislike", "angry"]
    
    text = text.lower()
    pos_count = sum([1 for word in positive_words if word in text])
    neg_count = sum([1 for word in negative_words if word in text])
    
    if pos_count > neg_count:
        # Format: [negative, positive]
        probs = [0.2, 0.8]
        sentiment = 1  # positive
    elif neg_count > pos_count:
        probs = [0.8, 0.2]
        sentiment = 0  # negative
    else:
        probs = [0.5, 0.5]
        sentiment = 0  # default to negative with low confidence
    
    return {
        "sentiment": sentiment,
        "probabilities": probs,
        "confidence": max(probs)
    }

# Function to perform fusion prediction
def perform_fusion(bert_result, fer_result):
    # Prepare inputs for fusion model
    bert_tensor = torch.tensor([bert_result['probabilities']])
    fer_tensor = torch.tensor([fer_result['probabilities']])
    
    # Create fusion model
    fusion_model = late_fusion_model()
    
    # Get fusion prediction
    with torch.no_grad():
        fusion_output = fusion_model(bert_tensor, fer_tensor)
        prediction_idx = torch.argmax(fusion_output, dim=1).item()
        
    # Map prediction to sentiment
    sentiment_map = {
        0: "negative",
        1: "positive",
        2: "neutral/conflicting"
    }
    
    final_prediction = sentiment_map[prediction_idx]
    confidence = fusion_output[0, prediction_idx].item()
    
    return {
        "emotion": final_prediction,
        "confidence": confidence,
        "probabilities": fusion_output[0].tolist()
    }

def test_with_image_and_text(image_path, text):
    """Test the full pipeline with an image and text"""
    print("\n" + "="*50)
    print(f"Testing with image: {image_path}")
    print(f"Text: \"{text}\"")
    print("="*50)
    
    # Get FER prediction
    fer_result = predict_emotion_from_path(image_path)
    print("\nFER Result:")
    print(f"Emotion: {fer_result['emotion']}")
    print(f"Sentiment: {fer_result['sentiment']}")
    print(f"Confidence: {fer_result['confidence']:.2f}")
    
    # Get BERT prediction
    bert_result = mock_bert_sentiment(text)
    print("\nBERT Result:")
    print(f"Sentiment: {bert_result['sentiment']} ({'positive' if bert_result['sentiment'] == 1 else 'negative'})")
    print(f"Confidence: {bert_result['confidence']:.2f}")
    
    # Fusion prediction
    fusion_result = perform_fusion(bert_result, fer_result)
    
    print("\nFusion Result:")
    print(f"Final emotion: {fusion_result['emotion']}")
    print(f"Confidence: {fusion_result['confidence']:.2f}")
    
    return fer_result, bert_result, fusion_result

def main():
    # Example test image paths - update these with real paths if available
    # Otherwise, we'll use the webcam option
    test_cases = [
        {"image": "./test_images/happy.jpg", "text": "I'm feeling great today!"},
        {"image": "./test_images/sad.jpg", "text": "I'm really upset about this news."},
        {"image": "./test_images/neutral.jpg", "text": "I'm so disappointed with the results."}
    ]
    
    # If no test images, use webcam
    use_webcam = not all(os.path.exists(tc["image"]) for tc in test_cases)
    
    if use_webcam:
        print("No test images found. Please take a photo with the webcam.")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        for i, tc in enumerate(test_cases):
            print(f"\nTest case {i+1}: {tc['text']}")
            print("Press 'c' to capture an image...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                cv2.imshow('Press c to capture', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Process captured frame
                    fer_result = predict_emotion_from_array(frame)
                    bert_result = mock_bert_sentiment(tc['text'])
                    fusion_result = perform_fusion(bert_result, fer_result)
                    
                    print("\nFER Result:")
                    print(f"Emotion: {fer_result['emotion']}")
                    print(f"Sentiment: {fer_result['sentiment']}")
                    
                    print("\nBERT Result:")
                    print(f"Sentiment: {'positive' if bert_result['sentiment'] == 1 else 'negative'}")
                    
                    print("\nFusion Result:")
                    print(f"Final emotion: {fusion_result['emotion']}")
                    print(f"Confidence: {fusion_result['confidence']:.2f}")
                    
                    break
                
                elif key == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Test with provided images
        for tc in test_cases:
            test_with_image_and_text(tc["image"], tc["text"])

if __name__ == "__main__":
    main()