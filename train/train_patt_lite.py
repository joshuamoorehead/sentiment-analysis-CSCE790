import os
import numpy as np
import pandas as pd
import cv2
import h5py
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set parameters
NUM_CLASSES = 7
IMG_SHAPE = (120, 120, 3)
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = "./models/patt_lite.weights.h5"
DATA_DIR = "./data"
KAGGLE_DATASET = "deadskull7/fer2013"


# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Define PAtt-Lite model
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

        x = tf.expand_dims(x, axis=1)  # shape becomes (batch, 1, 32)
        x = self.self_attention([x, x])  # now valid input
        x = tf.squeeze(x, axis=1)        # back to (batch, 32)

        outputs = self.prediction_layer(x)
        return outputs
    
    def save_weights_to_file(self, filepath):
        self.save_weights(filepath)
    
    def load_weights_from_file(self, filepath):
        self.load_weights(filepath)
    
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

def download_kaggle_dataset():
    """Download the FER2013 dataset from Kaggle"""
    try:
        import kaggle
        print("Downloading FER2013 dataset from Kaggle...")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=DATA_DIR, unzip=True)
        print(f"Dataset downloaded to {DATA_DIR}")
        expected_csv = os.path.join(DATA_DIR, "fer2013.csv")
        if not os.path.exists(expected_csv):
            print("Dataset download complete, but fer2013.csv not found.")
            raise FileNotFoundError("fer2013.csv not found after unzip.")
        else:
            print("Dataset downloaded and verified.")
    except ImportError:
        print("Kaggle API not found. Please install it with: pip install kaggle")
        print("Also make sure you have your kaggle.json file in ~/.kaggle/")
        raise
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please make sure you have set up your Kaggle API credentials.")
        print("See: https://github.com/Kaggle/kaggle-api#api-credentials")
        raise

def process_fer2013_to_h5():
    """Process the raw FER2013 dataset and save to h5 file"""
    h5_path = os.path.join(DATA_DIR, "fer2013.h5")
    
    # Check if h5 file already exists
    if os.path.exists(h5_path):
        print(f"H5 file already exists at {h5_path}")
        return h5_path
    
    # Load CSV file
    csv_path = os.path.join(DATA_DIR, "fer2013.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return None
    
    print("Processing FER2013 dataset to h5 format...")
    data = pd.read_csv(csv_path)
    
    # Split data
    train_data = data[data['Usage'] == 'Training']
    val_data = data[data['Usage'] == 'PublicTest']
    test_data = data[data['Usage'] == 'PrivateTest']
    
    # Process images
    def process_batch(batch):
        images = []
        labels = []
        for _, row in tqdm(batch.iterrows(), total=len(batch)):
            pixels = np.array(row['pixels'].split(), dtype=np.uint8)
            image = pixels.reshape(48, 48)
            
            # Convert to RGB
            image_rgb = np.stack([image] * 3, axis=-1)
            
            # Resize to target size
            image_resized = cv2.resize(image_rgb, (IMG_SHAPE[0], IMG_SHAPE[1]))
            
            images.append(image_resized)
            labels.append(row['emotion'])
        
        return np.array(images), np.array(labels)
    
    print("Processing training data...")
    X_train, y_train = process_batch(train_data)
    
    print("Processing validation data...")
    X_val, y_val = process_batch(val_data)
    
    print("Processing test data...")
    X_test, y_test = process_batch(test_data)
    
    # Save to h5 file
    print(f"Saving to {h5_path}...")
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_val', data=X_val)
        f.create_dataset('y_val', data=y_val)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
    
    print("Processing complete.")
    return h5_path

def load_fer_dataset(dataset_path):
    """Load FER2013 dataset from h5 file"""
    with h5py.File(dataset_path, 'r') as f:
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        X_val = f['X_val'][:]
        y_val = f['y_val'][:]
        X_test = f['X_test'][:]
        y_test = f['y_test'][:]
    
    # Shuffle training data
    X_train, y_train = shuffle(X_train, y_train)
    
    print(f"Dataset loaded: {X_train.shape}, {y_train.shape}, {X_val.shape}, {y_val.shape}, {X_test.shape}, {y_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('./models/training_history.png')
    plt.show()

def main():
    # Download and process dataset
    download_kaggle_dataset()
    h5_path = process_fer2013_to_h5()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_fer_dataset(h5_path)
    
    # Create model
    model = PAttLite(num_classes=NUM_CLASSES, img_shape=IMG_SHAPE)
    model.build((None, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2]))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback, early_stopping]
    )
    
    model.summary()
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save final weights
    model.save_weights_to_file(CHECKPOINT_PATH)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history

if __name__ == "__main__":
    main()