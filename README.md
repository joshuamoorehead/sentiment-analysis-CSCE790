# Facial Expression Recognition at the Edge

1. Facial Expression Recognition (FER) - PAtt-Lite Model
2. Sentiment Analysis - BERT
3. Data Fusion of FER and Sentiment Analysis Networks

## Usage on Demo Raspberry Pi
```
source rwvenv_coral/bin/activate
python main.py
```

## Setup to Use with Coral TPU
From [Coral docs](https://coral.ai/docs/accelerator/get-started/#runtime-on-linux):
```
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

sudo apt-get update
sudo apt-get install libedgetpu1-std
```

Install Python version 3.9:
```
pyenv install -v 3.9.0
```

Create Python virtual environment with pyenv Python version:
```
pyenv/versions/3.9.0/bin/python -m venv rwvenv_coral
```

Add Python version to PATH:
```
PATH="/home/mreza/.pyenv/versions/3.9.0/bin/:$PATH"
```

Install [PyCoral](https://coral.ai/docs/accelerator/get-started/#2-install-the-pycoral-library):
WARN: this didn't work
```
sudo apt-get install python3-pycoral
```

Install PyCoral using Pip ((recommended)[https://stackoverflow.com/questions/77897444/pycoral-but-it-is-not-going-to-be-installed]):
```
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

---

## 1. Facial Expression Recognition (FER)

We use a lightweight PAtt-Lite convolutional neural network trained on the FER2013 dataset for expression recognition. The model is designed for edge inference on devices like the Raspberry Pi 5.

- Training script: `train/train_patt_lite.py`
- Model architecture: `models/patt_lite.py`
- Inference script: `infer_patt_lite.py`



## 2. Sentiment Analysis (BERT)

The sentiment analysis is done using a fine-tuned BERT model with a custom classification head.

- Training notebook: `train/bertforsentimentanalysis.ipynb`
- Model architecture: `models/bert.py`
- Inference script `run_sentiment_analysis.py`

We fine-tuned BERT on sentiment analysis datasets including Archeage, Ntua, and HCR.

---

## 3. Multimodal Fusion

The FER and sentiment models are combined in `multimodalfusion.py` to create a joint expression/sentiment decision model. Future iterations aim to improve fusion latency and decision thresholds for real-time applications.

---

## Install

Developed for Raspberry Pi 5:

```bash
sudo apt install portaudio19-dev
sudo apt install flac
```

---

## Sources

* https://raspberrypi.stackexchange.com/questions/84666/problem-on-installing-pyaudio-on-raspberry-pi
* https://github.com/kpolley/Python_AVrecorder/blob/master/picam.py
* https://github.com/JLREX/PAtt-Lite
