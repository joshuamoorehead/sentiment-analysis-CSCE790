import os
import time
import threading
from mic_driver import SpeechToText
from cam_driver import CaptureVideo
from infer_patt_lite import predict_emotion_from_path 
from run_sentiment_analysis import SentimentAnalysis  
from multimodalfusion1 import late_fusion_model  

def main():
    # Init save directory
    save_dir = './tmp'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    # Initialize camera and microphone
    camera = CaptureVideo(save_dir, DEBUG=True)
    mic = SpeechToText(save_dir, DEBUG=True)
    
    # Capture image and speech
    print('Readying camera and microphone...')
    time.sleep(2)
    
    filename = str(time.time())
    camera.start(filename + ".png")
    mic.start(filename + ".txt")  # Start speech recording
    
    # Wait for capture to complete (camera takes 10 frames * 2 seconds each)
    print("Capturing image and speech...")
    time.sleep(25)  # Give plenty of time for both to complete
    
    # Get file paths - camera saves multiple frames with numbered prefixes
    img_filename = "0_" + filename + ".png"  # Use first frame
    image_path = os.path.join(save_dir + "/vid/", img_filename)  # Camera adds /vid/ internally
    text_path = os.path.join(save_dir, filename + ".txt")  # Path for text file
    
    # Wait for files to exist
    max_wait = 10
    waited = 0
    while not (os.path.exists(image_path) and os.path.exists(text_path)):  # Wait for both files
        time.sleep(0.5)
        waited += 0.5
        if waited >= max_wait:
            print(f"Timeout waiting for files. Image: {os.path.exists(image_path)}, Text: {os.path.exists(text_path)}")
            return
    
    # Process the captured image (FER)
    print("Analyzing facial expression...")
    fer_result = predict_emotion_from_path(image_path)
    
    # Display FER result
    print("\n=== Facial Expression Analysis ===")
    print(f"Detected emotion: {fer_result['emotion']}")
    print(f"Sentiment: {fer_result['sentiment']} (-1=negative, 0=neutral, 1=positive)")
    print(f"Confidence: {fer_result['confidence']:.2f}")
    
    # Process the captured speech (BERT)
    print("\nAnalyzing speech sentiment...")
    # Read text from file
    try:
        with open(text_path, 'r') as f:
            text = f.read().strip()
            if not text:
                print("Text file is empty. Speech recognition may have failed.")
                return
    except FileNotFoundError:
        print(f"Text file not found at {text_path}")
        return
    
    # Initialize BERT model
    sentiment_analyzer = SentimentAnalysis(debug=True)  # Set debug to see outputs
    bert_model = sentiment_analyzer.load_model('./weights/bert_model.pt')
    
    # Get BERT prediction
    bert_prediction = sentiment_analyzer.inference(bert_model, text)
    
    # BERT model returns a tensor with the class index (0 or 1)
    # Here we convert it to the format needed by the fusion model
    bert_sentiment = bert_prediction.item()
    bert_probabilities = [0.9, 0.1] if bert_sentiment == 0 else [0.1, 0.9]
    
    bert_result = {
        "sentiment": bert_sentiment,
        "probabilities": bert_probabilities
    }
    
    # Display BERT result
    print("\n=== Speech Sentiment Analysis ===")
    print(f"Text: '{text}'")
    print(f"Sentiment: {'Positive' if bert_result['sentiment'] == 1 else 'Negative'}")
    
    # Now combine using fusion model
    print("\nPerforming multimodal fusion...")
    # Prepare inputs for fusion model
    import torch
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
    
    # Display final result
    print("\n=== Final Emotion Analysis ===")
    print(f"Facial Expression: {fer_result['emotion']}")
    print(f"Speech Sentiment: {'Positive' if bert_result['sentiment'] == 1 else 'Negative'}")
    print(f"Combined Emotion: {final_prediction} (confidence: {confidence:.2f})")
    
    return {
        "fer_result": fer_result,
        "bert_result": bert_result,
        "final_prediction": final_prediction,
        "confidence": confidence
    }

if __name__ == "__main__":
    main()