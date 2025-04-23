"""
Convert PAttLite model to TensorFlow Lite format
Run this on a system where TensorFlow works properly, then copy the .tflite file to your Raspberry Pi
"""
import os
import sys
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

def convert_to_tflite(weights_path="weights/patt_lite.weights.h5", output_path="weights/patt_lite_model.tflite"):
    """Convert PAttLite model to TensorFlow Lite format"""
    try:
        # Import PAttLite model
        print("Importing PAttLite model...")
        from models.patt_lite import PAttLite
        
        # Check if weights file exists
        if not os.path.exists(weights_path):
            print(f"Error: Weights file not found at {weights_path}")
            return False
            
        # Load the model
        print(f"Loading model weights from {weights_path}")
        model = PAttLite()
        model.build((None, 120, 120, 3))
        model.load_weights_from_file(weights_path)
        
        # Run a test prediction to ensure the model is fully built
        print("Running test prediction to initialize model...")
        import numpy as np
        test_input = np.random.random((1, 120, 120, 3)).astype(np.float32)
        test_output = model(test_input, training=False)
        print(f"Test prediction shape: {test_output.shape}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to TFLite
        print("Converting model to TFLite format...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the model
        print(f"Saving TFLite model to {output_path}")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"Conversion complete! TFLite model saved to {output_path}")
        model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error during model conversion: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get optional custom paths from command line arguments
    weights_path = "weights/patt_lite.weights.h5"
    output_path = "weights/patt_lite_model.tflite"
    
    if len(sys.argv) > 1:
        weights_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        
    convert_to_tflite(weights_path, output_path)