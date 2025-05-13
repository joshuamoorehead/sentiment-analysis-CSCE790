# Multi-Modal Emotion Recognition at the Edge

This project implements a real-time, multi-modal emotion recognition system designed for deployment on resource-constrained devices like the Raspberry Pi 5. It fuses facial expression recognition (FER) and sentiment analysis (text) to classify emotions as positive, neutral, or negative.

Developed as a final research project for CSCE 790 at the University of South Carolina.

## Features

- Lightweight facial expression recognition using a custom PAtt-Lite CNN architecture
- BERT-based sentiment analysis with fine-tuned binary classification head
- Late-decision data fusion strategy combining FER and sentiment signals
- Edge deployment support on Raspberry Pi 5 with Coral TPU compatibility
- Live webcam and microphone input interface
- Modular design for future support of tri-modal (audio/text/visual) analysis

## My Contributions

- Implemented the entire FER pipeline using PAtt-Lite
- Designed and trained the FER model on FER2013, optimized for edge performance
- Developed Raspberry Pi camera integration for live FER inference
- Integrated FER into the late-decision fusion model
- Conducted inference speed and accuracy benchmarking

## Quickstart (Demo on Raspberry Pi)

```bash
source rwvenv_coral/bin/activate
python main.py
```

## Setup: Coral TPU Support (Optional)

```bash
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

Install Python 3.9 and create a virtual environment:

```bash
pyenv install -v 3.9.0
pyenv/versions/3.9.0/bin/python -m venv rwvenv_coral
```

Activate environment and install PyCoral (if using Coral):

```bash
source rwvenv_coral/bin/activate
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

## Project Structure

- `models/patt_lite.py` – PAtt-Lite model definition
- `train/train_patt_lite.py` – FER training pipeline
- `run_sentiment_analysis.py` – BERT-based sentiment inference
- `multimodalfusion.py` – Late-decision fusion logic
- `main.py` – Entry point for Raspberry Pi emotion recognition

## Dataset Details

- **FER2013** – Used for training the FER model
- **Archeage, Ntua, HCR, IMDB** – Used for fine-tuning the sentiment model

## Performance

- FER model: ~62.15% test accuracy on FER2013, 1.5s inference time on Raspberry Pi
- Sentiment model: ~93.01% test accuracy, ~313ms inference time
- Combined system supports real-time operation and produces improved contextual emotion predictions

## Future Work

- Improve FER model accuracy using curriculum learning or pretrained ViTs
- Implement speech-based sentiment (prosody, tone) for tri-modal fusion
- Replace current late-fusion logic with a trained fusion layer
- Deploy FER model to Coral TPU for sub-second performance

## Authors

- **Joshua Moorehead** – Facial expression recognition, Pi deployment, FER integration
- **Haley Lind** – Multimodal fusion strategy, fusion scripting, literature review
- **Regan Willis** – Sentiment model fine-tuning, speech-to-text system